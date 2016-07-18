"""Condenses relevant data from the database into a few csv files.

This is horrible and long but it's also one-time use, so... ehhh.

nordkette
    Dumps Nordkette data to csv file.

cosmo7
    COSMO7 profiles stretched and interpolated to the retrieval grid.

raso
    Radiosonde profiles interpolated to the retrieval grid.

bt_igmk
    Dumps simulated brightness temperatures from IGMK (only the ones used in
    retrieval algorithms).

bt_monortm
    Simulates zenith BT with MonoRTM. source controls from where to get the
    profiles (None is database, else give a pattern for csv files). levels
    controls the number of layers that the profile is interpolated on (MonoRTM
    can only take 200 max, most sondes have more).

bt_mwrtm_fap
bt_mwrtm_full
    Simulates BT with MWRTM using either FAPs or the full absorption models.
    Calculates all angles contained in retrieval algorithms. levels controls
    the number of internal model levels, source is the same as for bt_monortm.
"""

import argparse
import datetime as dt
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from optimal_estimation import VirtualHATPRO, zgrid, z_hatpro
from db_tools import Database
import formulas as fml


def stretch(xs, lower=None, upper=None, power=1):
    """Part of the COSMO7 forecast postprocessing pipeline (stretches profile
    to reach actual surface height)."""
    n = len(xs)
    loffset = lower - xs[0] if lower is not None else 0.
    toffset = upper - xs[-1] if upper is not None else 0.
    out = xs + loffset * np.linspace(1, 0, n)**power
    out = out + toffset * np.linspace(0, 1, n)**power
    return out


def bt_dataset(data):
    """Unstacks elevation angles from hatpro database table."""
    angles = list(sorted(set(data.angle)))
    out = []
    for angle in angles:
        subset = (data
                .loc[(data.angle == angle) & (data.p < 970)]
                .drop_duplicates("valid")
                .set_index("valid")
                .drop(["angle", "rain", "kind", "id"], axis=1)
                )
        if angle == angles[0]:
            pTq = subset[["p", "T", "qvap"]]
        out.append(subset
                .drop(["p", "T", "qvap"], axis=1)
                .add_suffix("_{:0>4.1f}".format(angle))
                )
    out = pd.concat(out + [pTq], axis=1)
    out.index = out.index.map(dt.datetime.utcfromtimestamp)
    out.index.name = "valid"
    return out.round(10)


def get_raso_profiles(db):
    raso = db.as_dataframe("""
            SELECT profiles.valid, z, p, T, qvap, qliq
            FROM profiledata
            JOIN profiles ON profiles.id = profiledata.profile
            WHERE profiles.kind = "raso" AND z > 500
            ORDER BY z ASC, valid ASC;
            """)
    gvalids = raso.groupby("valid")
    for valid in gvalids.groups:
        df = (gvalids
                .get_group(valid)
                .dropna(axis=0)
                .drop_duplicates("z", keep="first")
                )
        zs = df["z"].values
        if len(zs) < 50 or zs[0] > 612 or zs[-1] < 15000: continue
        # Interpolate to 612 m level at the bottom
        zs = df["z"].values
        surface = OrderedDict()
        for col in df:
            surface[col] = [float(interp1d(zs, df[col].values)(z_hatpro))]
        surface = pd.DataFrame.from_dict(surface)
        # Remove values below surface and add surface
        valid = dt.datetime.utcfromtimestamp(valid)
        out = pd.concat([surface, df.loc[df["z"]>z_hatpro]], axis=0)
        yield valid, out.reset_index(drop=True)


def get_csv_profiles(filename):
    assert "<VAR>" in filename
    def reader(var):
        fn = filename.replace("<VAR>", var)
        return pd.read_csv(fn, parse_dates=["valid"], index_col="valid")
    pdf = reader("p")
    Tdf = reader("T").reindex(pdf.index)
    qvapdf = reader("qvap").reindex(pdf.index)
    qliqdf = reader("qliq").reindex(pdf.index)
    z = pd.Series(zgrid, name="z")
    for (vp, ps), (vT, Ts), (vv, qvaps), (vl, qliqs) in zip(pdf.iterrows(),
            Tdf.iterrows(), qvapdf.iterrows(), qliqdf.iterrows()):
        assert vp == vT == vv == vl
        p = pd.Series(ps.values, name="p")
        T = pd.Series(Ts.values, name="T")
        qvap = pd.Series(qvaps.values, name="qvap")
        qliq = pd.Series(qliqs.values, name="qliq")
        yield vp, pd.concat([z, p, T, qvap, qliq], axis=1)


def make_lnq(data=None, qvap=None, qliq=None):
    if data is not None:
        assert qvap is None and qliq is None
        lnq = pd.Series(np.log(data["qvap"] + data["qliq"]),
                index=data.index, name="lnq")
        lnq[lnq<-30] = -30 # remove -infs
        return lnq
    else:
        out = np.log(qvap + qliq)
        out[out<-30] = -30
        return out


parser = argparse.ArgumentParser()
parser.add_argument("--levels", default=3000, type=int)
parser.add_argument("--source", default=None, type=str)
parser.add_argument("data")

if __name__ == "__main__":

    args = parser.parse_args()

    db = Database("../data/amalg.db")


    if args.data == "nordkette":
        df = db.as_dataframe("SELECT * FROM nordkette ORDER BY valid ASC;").set_index("valid")
        df.index = df.index.map(dt.datetime.utcfromtimestamp)
        df.columns = [c.replace("T_", "z=") for c in df.columns]
        df.index.name = "valid"
        df.to_csv("../data/unified/T_nordkette.csv")


    if args.data == "cosmo7":
        # COSMO7 profiles interpolated to retrieval grid
        cosmo7 = db.as_dataframe("""
                SELECT profiles.valid, profiles.lead, z, p, T, qvap, qliq
                FROM profiledata
                JOIN profiles ON profiles.id = profiledata.profile
                WHERE profiles.kind = "cosmo7" AND lead <= 108000
                ORDER BY z ASC;
                """)

        gleads = cosmo7.groupby("lead")
        for glead in gleads.groups:
            data = gleads.get_group(glead)
            lead = set(data["lead"]).pop()
            gvalids = data.drop(["lead"], axis=1).groupby("valid")
            valid = []
            out = {"p": [], "T": [], "qvap": [], "qliq": []}
            for gvalid in gvalids.groups:
                profile = gvalids.get_group(gvalid)
                valid.append(dt.datetime.utcfromtimestamp(set(profile["valid"]).pop()))
                # Stretch, then interpolate
                zs = stretch(profile["z"].values, lower=z_hatpro, power=1)
                for var in out:
                    out[var].append(interp1d(zs, profile[var].values)(zgrid))
            valid = pd.Series(valid, name="valid")
            for var, values in out.items():
                (pd.DataFrame(np.vstack(values), index=valid,
                        columns=["z={}m".format(int(z)) for z in zgrid])
                        .sort_index()
                        .round(10)
                        .to_csv("../data/unified/{}_cosmo7+{:0>2}.csv".format(var, lead//3600))
                        )


    if args.data == "raso":
        # Radiosoundings interpolated to retrieval grid

        raso = db.as_dataframe("""
                SELECT profiles.valid, z, p, T, qvap, qliq
                FROM profiledata
                JOIN profiles ON profiles.id = profiledata.profile
                WHERE profiles.kind = "raso" AND z > 500
                ORDER BY z ASC, valid ASC;
                """)

        valid = []
        out = {"p": [], "T": [], "qvap": [], "qliq": []}
        gvalids = raso.groupby("valid")
        for gvalid in gvalids.groups:
            profile = gvalids.get_group(gvalid).dropna(axis=0).drop_duplicates("z", keep="first")
            zs = profile["z"].values
            if len(zs) < 50 or zs[0] > 612 or zs[-1] < 15000: continue
            valid.append(dt.datetime.utcfromtimestamp(set(profile["valid"]).pop()))
            for var in out:
                out[var].append(interp1d(zs, profile[var].values)(zgrid))
        valid = pd.Series(valid, name="valid")
        for var, values in out.items():
            (pd.DataFrame(np.vstack(values), index=valid,
                    columns=["z={}m".format(int(z)) for z in zgrid])
                    .sort_index()
                    .round(10)
                    .to_csv("../data/unified/{}_raso.csv".format(var))
                    )


    if args.data == "bt_igmk":
        # IGMK processed brightness temperatures
        igmk = db.as_dataframe("""
                SELECT * FROM hatpro
                WHERE kind = "igmk"
                ORDER BY valid ASC;
                """)
        columns = ["TB_22240MHz_00.0", "TB_23040MHz_00.0",
                "TB_23840MHz_00.0", "TB_25440MHz_00.0", "TB_26240MHz_00.0",
                "TB_27840MHz_00.0", "TB_31400MHz_00.0", "TB_51260MHz_00.0",
                "TB_52280MHz_00.0", "TB_53860MHz_00.0", "TB_54940MHz_00.0",
                "TB_54940MHz_60.0", "TB_54940MHz_70.8", "TB_54940MHz_75.6",
                "TB_54940MHz_78.6", "TB_54940MHz_81.6", "TB_54940MHz_83.4",
                "TB_54940MHz_84.6", "TB_54940MHz_85.2", "TB_54940MHz_85.8",
                "TB_56660MHz_00.0", "TB_56660MHz_60.0", "TB_56660MHz_70.8",
                "TB_56660MHz_75.6", "TB_56660MHz_78.6", "TB_56660MHz_81.6",
                "TB_56660MHz_83.4", "TB_56660MHz_84.6", "TB_56660MHz_85.2",
                "TB_56660MHz_85.8", "TB_57300MHz_00.0", "TB_57300MHz_60.0",
                "TB_57300MHz_70.8", "TB_57300MHz_75.6", "TB_57300MHz_78.6",
                "TB_57300MHz_81.6", "TB_57300MHz_83.4", "TB_57300MHz_84.6",
                "TB_57300MHz_85.2", "TB_57300MHz_85.8", "TB_58000MHz_00.0",
                "TB_58000MHz_60.0", "TB_58000MHz_70.8", "TB_58000MHz_75.6",
                "TB_58000MHz_78.6", "TB_58000MHz_81.6", "TB_58000MHz_83.4",
                "TB_58000MHz_84.6", "TB_58000MHz_85.2", "TB_58000MHz_85.8",
                "p", "T", "qvap"]
        bt_dataset(igmk)[columns].sort_index().to_csv("../data/unified/bt_igmk.csv")


    if args.data == "bt_monortm":
        # MonoRTM brightness temperatures (zenith only)
        from monortm import MonoRTM
        from monortm.hatpro import config
        from monortm.profiles import from_mwrt_profile

        if args.source is None:
            data = get_raso_profiles(db)
            srcname = "_hr"
        else:
            data = get_csv_profiles(args.source)
            srcname = ""

        levels = min(args.levels, 180)
        profiles, valids = [], []
        ps, Ts, qs = [], [], []
        for valid, df in data:
            if len(df) > levels:
                z = np.logspace(np.log10(z_hatpro), np.log10(15000.), levels)
                p = interp1d(df["z"].values, df["p"].values)(z)
                T = interp1d(df["z"].values, df["T"].values)(z)
                qvap = interp1d(df["z"].values, df["qvap"].values)(z)
                qliq = interp1d(df["z"].values, df["qliq"].values)(z)
            else:
                z = df["z"].values
                p = df["p"].values
                T = df["T"].values
                qvap = df["qvap"].values
                qliq = df["qliq"].values
            lnq = make_lnq(qvap=qvap, qliq=qliq)
            profiles.append(from_mwrt_profile(z, p, T, lnq))
            valids.append(valid)
            ps.append(p[0])
            Ts.append(T[0])
            qs.append(qvap[0])
        bts = MonoRTM.run_distributed(config, profiles, get="brightness_temperatures")
        valids = pd.Series(valids, name="valid")
        columns = [
                "TB_22240MHz_00.0", "TB_23040MHz_00.0", "TB_23840MHz_00.0",
                "TB_25440MHz_00.0", "TB_26240MHz_00.0", "TB_27840MHz_00.0",
                "TB_31400MHz_00.0", "TB_51260MHz_00.0", "TB_52280MHz_00.0",
                "TB_53860MHz_00.0", "TB_54940MHz_00.0", "TB_56660MHz_00.0",
                "TB_57300MHz_00.0", "TB_58000MHz_00.0"
                ]
        df = pd.DataFrame(np.vstack(bts), columns=columns, index=valids)
        ps = pd.Series(ps, name="p", index=valids)
        Ts = pd.Series(Ts, name="T", index=valids)
        qs = pd.Series(qs, name="qvap", index=valids)
        (pd.concat([df, ps, Ts, qs], axis=1)
                .sort_index()
                .to_csv("../data/unified/bt_monortm{}.csv".format(srcname))
                )


    if args.data.startswith("bt_mwrtm"):
        # MWRTM brightness temperatures
        from mwrt import MWRTM, LinearInterpolation
        if args.data.endswith("_fap"):
            from faps_hatpro import *
            absname = "fap"
            absorp = [
                    FAP22240MHz, FAP23040MHz, FAP23840MHz, FAP25440MHz,
                    FAP26240MHz, FAP27840MHz, FAP31400MHz, FAP51260MHz,
                    FAP52280MHz, FAP53860MHz, FAP54940MHz, FAP56660MHz,
                    FAP57300MHz, FAP58000MHz
                    ]
        elif args.data.endswith("_full"):
            from mwrt.fapgen import absorption_model
            from mwrt import tkc, liebe93
            absname = "full"
            absmod = absorption_model(liebe93.refractivity_gaseous, tkc.refractivity_lwc)
            absorp = [
                    partial(absmod, 22.240), partial(absmod, 23.040),
                    partial(absmod, 23.840), partial(absmod, 25.440),
                    partial(absmod, 26.240), partial(absmod, 27.840),
                    partial(absmod, 31.400), partial(absmod, 51.260),
                    partial(absmod, 52.280), partial(absmod, 53.860),
                    partial(absmod, 54.940), partial(absmod, 56.660),
                    partial(absmod, 57.300), partial(absmod, 58.000)
                    ]
        else: raise ValueError("unknown absorption model")
        tb_names = ["TB_22240MHz", "TB_23040MHz", "TB_23840MHz", "TB_25440MHz",
                    "TB_26240MHz", "TB_27840MHz", "TB_31400MHz", "TB_51260MHz",
                    "TB_52280MHz", "TB_53860MHz", "TB_54940MHz", "TB_56660MHz",
                    "TB_57300MHz", "TB_58000MHz"]
        angles = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8]]
        columns = ["{}_{:0>4.1f}".format(f, a)
                for f, angs in zip(tb_names, angles)
                for a in angs] + ["p", "T", "qvap"]

        if args.source is None:
            data = get_raso_profiles(db)
            srcname = "_hr"
        else:
            data = get_csv_profiles(args.source)
            srcname = ""

        zmodel = np.logspace(np.log10(z_hatpro), np.log10(15000.), args.levels)
        valids, rows = [], []
        for valid, df in data:
            bts = []
            itp = LinearInterpolation(source=df["z"].values, target=zmodel)
            for ap, ang in zip(absorp, angles):
                model = MWRTM(itp, ap)
                bts.extend(model.forward(angles=ang, data=df, lnq=make_lnq(data=df)))
            row = bts + [df["p"].values[0], df["T"].values[0], df["qvap"].values[0]]
            rows.append(row)
            valids.append(valid)
        outname = "../data/unified/bt_mwrtm_{}_{}{}.csv".format(args.levels, absname, srcname)
        valids = pd.Series(valids, name="valid")
        pd.DataFrame(rows, index=valids, columns=columns).sort_index().to_csv(outname)

