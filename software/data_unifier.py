"""Data pipeline step 2: condense data of interest into a few csv files.

Reads data from the database created by db_import.py and writes it into
easier-to-read csv files. Utilizes MWRTM and MonoRTM to calculate brightness
temperatures. Interpolates profiles to the retrieval grid. Can also combine
data from previously created csv files (see get_csv_profiles function). Like
db_import.py this is a one-time use script and rather ugly, long and without
much optimization. :(

"python data_unifier.py --help" for information on the command line arguments.
Available values for the data argument:

nordkette
    Nordkette temperature time series.

cosmo7
    COSMO7 profiles stretched and interpolated to the retrieval grid.

raso
    Radiosonde profiles interpolated to the retrieval grid.

tb_igmk
    Dumps simulated brightness temperatures from IGMK (only the ones used in
    retrieval algorithms).

cloudy_igmk
    Cloud yes/no based on IGMK-processed data.

tb_monortm
    Simulates zenith BT with MonoRTM. source controls from where to get the
    profiles (None is database, else give a pattern for csv files). levels
    controls the number of layers that the profile is interpolated on (MonoRTM
    can only take 200 max, most sondes have more).

tb_mwrtm_fap
tb_mwrtm_full
    Simulates BT with MWRTM using either FAPs or the full absorption models.
    Calculates all angles contained in retrieval algorithms. levels controls
    the number of internal model levels, source is the same as for tb_monortm.
"""

import argparse
import datetime as dt
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from optimal_estimation import VirtualHATPRO, rgrid, mgrid, z_hatpro, z_top
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


def tb_dataset(data):
    """Unstacks elevation angles from hatpro database table."""
    angles = list(sorted(set(data.angle)))
    out = []
    for angle in angles:
        subset = (data
                .loc[(data.angle == angle) & (data.p < 970)]
                .drop_duplicates("valid")
                .set_index("valid")
                .drop(["angle", "kind", "id"], axis=1)
                )
        if angle == angles[0]:
            pTqr = subset[["p", "T", "qvap", "rain"]]
        out.append(subset
                .drop(["p", "T", "qvap", "rain"], axis=1)
                .add_suffix("_{:0>4.1f}".format(angle))
                )
    out = pd.concat(out + [pTqr], axis=1)
    out.index = out.index.map(dt.datetime.utcfromtimestamp)
    out.index.name = "valid"
    # Database stores 64 bit values, original data were 32 bit, rounding
    # removes all the ...9999s and ...0001s and reduces the csv file size
    return out.round(10)


def get_raso_profiles(db):
    """Iterate over radiosonde profiles from the database."""
    # Load all data and use pandas' groupby operation to separate valid times
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
        # Simple quality control:
        if len(zs) < 50 or zs[0] > z_hatpro or zs[-1] < z_top: continue
        # Interpolate to 612 m level at the bottom
        zs = df["z"].values
        surface = OrderedDict()
        for col in df:
            surface[col] = [float(interp1d(zs, df[col].values)(z_hatpro))]
        surface = pd.DataFrame.from_dict(surface)
        # Remove values below surface and add surface
        valid = dt.datetime.utcfromtimestamp(valid)
        out = pd.concat([surface, df.loc[df["z"]>z_hatpro]], axis=0)
        # Index needs to be reset (still has index from database query)
        yield valid, out.reset_index(drop=True)


def get_csv_profiles(filename):
    """Create dataframes compatible to those of get_raso_profiles by combining
    data from csv files each containing one of the variables.
    
    The filename must have a "<VAR>" substring which is a placeholder for each
    variable p, T, qvap and qliq. These four files are read and the function
    generates dataframes.

    /!\ The csv files must have aligned valid datetimes else an AssertionError
        will occur.
    """
    assert "<VAR>" in filename
    def reader(var):
        fn = filename.replace("<VAR>", var)
        return pd.read_csv(fn, parse_dates=["valid"], index_col="valid")
    pdf = reader("p")
    Tdf = reader("T").reindex(pdf.index)
    qvapdf = reader("qvap").reindex(pdf.index)
    qliqdf = reader("qliq").reindex(pdf.index)
    z = pd.Series(rgrid, name="z")
    for (vp, ps), (vT, Ts), (vv, qvaps), (vl, qliqs) in zip(pdf.iterrows(),
            Tdf.iterrows(), qvapdf.iterrows(), qliqdf.iterrows()):
        # Check that valid datetimes match
        assert vp == vT == vv == vl
        # Concatenate each variable into a dataframe, add height grid and yield
        p = pd.Series(ps.values, name="p")
        T = pd.Series(Ts.values, name="T")
        qvap = pd.Series(qvaps.values, name="qvap")
        qliq = pd.Series(qliqs.values, name="qliq")
        yield vp, pd.concat([z, p, T, qvap, qliq], axis=1)


def make_lnq(data=None, qvap=None, qliq=None):
    """lnq maker that handles 0 water content and limits lnq to values > -30.
    
    Function either takes a dataframe containing qvap and qliq columns with
    data argument or these columns separately with the corresponding arguments.
    """
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


# Exported brightness temperature columns
tb_columns = ["TB_22240MHz_00.0", "TB_23040MHz_00.0",
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
        "TB_58000MHz_84.6", "TB_58000MHz_85.2", "TB_58000MHz_85.8"]


parser = argparse.ArgumentParser()
parser.add_argument("--levels", default=3000, type=int, help="""
        Number of levels used for the vertical discretization in the RTM.
        Ignored if no radiative transfer calculations are performed.
        """)
parser.add_argument("--source", default=None, type=str, help="""
        All input data are taken from the database by default. A set of csv
        files to be used instead can be specifed with this argument. See
        get_csv_files function.
        """)
parser.add_argument("data", help="""
        What kind of data are processed and how?
        """)

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
        # Select only lead times up to +30
        cosmo7 = db.as_dataframe("""
                SELECT profiles.valid, profiles.lead, z, p, T, qvap, qliq
                FROM profiledata
                JOIN profiles ON profiles.id = profiledata.profile
                WHERE profiles.kind = "cosmo7" AND lead <= 108000
                ORDER BY z ASC;
                """)
        # Each lead time gets its own set of output files
        gleads = cosmo7.groupby("lead")
        for glead in gleads.groups:
            data = gleads.get_group(glead)
            lead = set(data["lead"]).pop()
            gvalids = data.drop(["lead"], axis=1).groupby("valid")
            # Temporary containers for data
            valid = []
            out = {"p": [], "T": [], "qvap": [], "qliq": []}
            for gvalid in gvalids.groups:
                profile = gvalids.get_group(gvalid)
                valid.append(dt.datetime.utcfromtimestamp(set(profile["valid"]).pop()))
                # Stretch, then interpolate to retrieval grid
                zs = stretch(profile["z"].values, lower=z_hatpro, power=1)
                for var in out:
                    out[var].append(interp1d(zs, profile[var].values)(rgrid))
            valid = pd.Series(valid, name="valid")
            for var, values in out.items():
                (pd.DataFrame(np.vstack(values), index=valid,
                        columns=["z={}m".format(int(z)) for z in rgrid])
                        .sort_index()
                        .round(10)
                        .to_csv("../data/unified/{}_cosmo7+{:0>2}.csv".format(var, lead//3600))
                        )


    if args.data == "raso":
        # Basically the same as cosmo7 but for radiosonde data
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
            if len(zs) < 50 or zs[0] > z_hatpro or zs[-1] < z_top: continue
            valid.append(dt.datetime.utcfromtimestamp(set(profile["valid"]).pop()))
            for var in out:
                out[var].append(interp1d(zs, profile[var].values)(rgrid))
        valid = pd.Series(valid, name="valid")
        for var, values in out.items():
            (pd.DataFrame(np.vstack(values), index=valid,
                    columns=["z={}m".format(int(z)) for z in rgrid])
                    .sort_index()
                    .round(10)
                    .to_csv("../data/unified/{}_raso.csv".format(var))
                    )


    if args.data == "tb_igmk":
        # IGMK processed brightness temperatures
        igmk = db.as_dataframe("""
                SELECT * FROM hatpro
                WHERE kind = "igmk"
                ORDER BY valid ASC;
                """)
        tb_dataset(igmk)[tb_columns+["p", "T", "qvap"]].sort_index().to_csv("../data/unified/TB_igmk.csv")


    if args.data == "tb_hatpro":
        # Dump HATPRO data from the database after filtering out all unwanted
        # columns and rows
        df = db.as_dataframe("""
                SELECT * FROM hatpro
                WHERE kind = "hatpro_blb"
                ORDER BY valid ASC;
                """)
        df = tb_dataset(df)
        tbs = df[tb_columns]
        sfc = df[["p", "T", "qvap", "rain"]]
        # Primitive quality control
        tbs = tbs.where(df>0).dropna()
        # Scanning frequency changed after 08-05 and data seem inconsistent after 10-29
        tbs = tbs = tbs.loc[((tbs.index > "2015-08-05 07:50:41") & (tbs.index < "2015-10-29 02:15:06")),:]
        sfc = sfc.ix[tbs.index]
        tbs.to_csv("../data/unified/test/TB_hatpro.csv")
        sfc.to_csv("../data/unified/test/sfc_hatpro.csv")


    if args.data == "cloudy_igmk":
        # IGMK has different cloud determination
        import xarray
        from glob import glob
        def read_netcdf(file):
            """Get cloud information from original NetCDF files of IGMK data."""
            xdata = xarray.open_dataset(file)
            xdata.coords["n_date"] = xdata.data_vars["date"]
            def to_date(d):
                return (dt.datetime
                        .strptime(str(d), "%Y%m%d%H")
                        )
            df = xdata["cloud_base"].to_dataframe()
            df = df.ix[~df.index.duplicated(keep="first")]
            df = (df.unstack(level=[1,2]) > -1).any(axis=1)
            df.index = df.index.map(to_date)
            df.index.name = "valid"
            df.name = "cloudy"
            return df.to_frame()
        pd.concat([read_netcdf(file)
                for file in sorted(glob("../data/hatpro_netcdf/rt*.cdf"))],
                axis=0).sort_index().to_csv("../data/unified/cloudy_igmk.csv")


    if args.data == "tb_monortm":
        # MonoRTM brightness temperatures (zenith only)
        from monortm import MonoRTM
        from monortm.hatpro import config
        from monortm.profiles import from_mwrt_profile

        # Data source switch
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
                zs = df["z"].values
                z = np.logspace(np.log10(z_hatpro), np.log10(zs[-1]-0.001), levels)
                p = interp1d(zs, df["p"].values)(z)
                T = interp1d(zs, df["T"].values)(z)
                qvap = interp1d(zs, df["qvap"].values)(z)
                qliq = interp1d(zs, df["qliq"].values)(z)
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
        tbs = MonoRTM.run_distributed(config, profiles, get="brightness_temperatures")
        valids = pd.Series(valids, name="valid")
        columns = [
                "TB_22240MHz_00.0", "TB_23040MHz_00.0", "TB_23840MHz_00.0",
                "TB_25440MHz_00.0", "TB_26240MHz_00.0", "TB_27840MHz_00.0",
                "TB_31400MHz_00.0", "TB_51260MHz_00.0", "TB_52280MHz_00.0",
                "TB_53860MHz_00.0", "TB_54940MHz_00.0", "TB_56660MHz_00.0",
                "TB_57300MHz_00.0", "TB_58000MHz_00.0"
                ]
        df = pd.DataFrame(np.vstack(tbs), columns=columns, index=valids)
        ps = pd.Series(ps, name="p", index=valids)
        Ts = pd.Series(Ts, name="T", index=valids)
        qs = pd.Series(qs, name="qvap", index=valids)
        (pd.concat([df, ps, Ts, qs], axis=1)
                .sort_index()
                .to_csv("../data/unified/TB_monortm{}.csv".format(srcname))
                )


    if args.data.startswith("tb_mwrtm"):
        # MWRTM brightness temperatures
        from mwrt import MWRTM, LinearInterpolation, tkc, liebe93
        from faps_hatpro import faps, tbs, bgs as bgs_fap, freqs
        from mwrt.fapgen import absorption_model, as_absorption
        from mwrt.background import USStandardBackground
        # Generate exact Liebe et al. 1993 absorption model
        absmod = absorption_model(liebe93.refractivity_gaseous, tkc.refractivity_lwc)
        absorp_l93 = [partial(absmod, f) for f in freqs]
        # Absorption model switch
        if args.data.endswith("_fap"):
            absname = "fap"
            absorp = faps
        elif args.data.endswith("_full"):
            absname = "full"
            absorp = absorp_l93
        else: raise ValueError("unknown absorption model")
        # MWRTM configuration
        angles = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8],
                [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8]]
        columns = ["{}_{:0>4.1f}".format(f, a)
                for f, angs in zip(tbs, angles)
                for a in angs] + ["p", "T", "qvap"]

        # Data source switch: create a funciton get_itp_and_bg (= get
        # interpolator and background) which returns the vertical
        # discretization interpolator and background brightness temperatures
        # used in the subsequent MWRTM calculations
        if args.source is None:
            # Full resolution profiles to max height
            data = get_raso_profiles(db)
            srcname = "_hr"
            def get_itp_and_bg(z):
                zmodel = np.logspace(np.log10(z_hatpro), np.log10(z[-1]), args.levels)
                itp = LinearInterpolation(z, zmodel)
                # Adapt background to acutal profile height instead of cutting
                # profile at some height
                usatm = USStandardBackground(lower=z[-1], upper=40000, n=1000)
                bgs = [usatm.evaluate(
                        partial(as_absorption(liebe93.refractivity_gaseous), f),
                        angle=0., cosmic=2.75) for f in freqs]
                return itp, bgs
        else:
            # Low resolution profiles, cut off at z_top
            data = get_csv_profiles(args.source)
            srcname = ""
            zmodel = np.logspace(np.log10(z_hatpro), np.log10(z_top), args.levels)
            def get_itp_and_bg(z):
                itp = LinearInterpolation(source=z, target=zmodel)
                return itp, bgs_fap

        valids, rows = [], []
        for valid, df in data:
            tbs = []
            itp, bgs = get_itp_and_bg(df["z"].values)
            for ap, ang, bg in zip(absorp, angles, bgs):
                model = MWRTM(itp, ap, background=bg)
                tbs.extend(model.forward(angles=ang, data=df, lnq=make_lnq(data=df)))
            row = tbs + [df["p"].values[0], df["T"].values[0], df["qvap"].values[0]]
            rows.append(row)
            valids.append(valid)
        outname = "../data/unified/TB_mwrtm_{}_{}{}.csv".format(args.levels, absname, srcname)
        valids = pd.Series(valids, name="valid")
        pd.DataFrame(rows, index=valids, columns=columns).sort_index().to_csv(outname)

