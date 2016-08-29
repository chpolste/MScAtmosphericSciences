"""Process data from some source and put it into a sqlite3 database.

A collection of one-time use data import scripts that assemble a unified
database from multiple data sources. To be used as a command line program from
the terminal (see the db_import bash script). This is not optimized for
efficiency and might take quite a bit of time and main memory to run. This
whole script isn't very pretty :(

size offset and pid arguments allow poor man's parallel processing by running
multiple instances of this script and afterwards merging all output databases
into the main one (done in the db_import bash script). pid governs the start
id for the produced rows, therefore the individual rows can be merged without
loosing appropriate relations between profiles and profiledata. size and offset
are used to select subsets of input files.

Available data sources:

raso_fwf
    Radiosonde data from fixed width format files. These make up the
    high-resolution climatology also used by Massaro (2013) and Meyer (2016).

raso_cosmo7
    Simulated vertical profiles from COSMO7 in L2E format.

raso_bufr
    Radiosonde data from bufr files. These are the additional profiles after
    2012 from ERTEL2.

nordkette
    Nordkette slope temperature measurements used previously by Meyer (2016).

igmk
    Brightness temperature simulations based on the high-resolution
    climatology. The used model is MONORTM, data were apparently processed by
    the IGMK. These data come from the netcdf files that are the input to the
    IDL script used by Massaro (2013) and Meyer (2016) for their regression
    retrievals.

hatpro
    HATPRO raw data import: BLB and BRT joined with data from the MET files.

description
    Adds a table information to the database containing descriptions of the
    kind values used in profiles and hatpro tables.
"""

import argparse, json, os
import datetime as dt
from glob import glob
from operator import itemgetter
from collections import OrderedDict

from toolz import groupby
import numpy as np
import pandas as pd

from db_tools import Database
import formulas as fml


scheme = (
        "CREATE TABLE IF NOT EXISTS profiles("
            "id INTEGER PRIMARY KEY, "
            "kind TEXT, "
            "valid NUMERIC, "
            "lead NUMERIC, "
            "cloudy INTEGER, "
            "file TEXT"
        "); "
        "CREATE TABLE IF NOT EXISTS profiledata("
            "id INTEGER PRIMARY KEY, "
            "profile INTEGER, "
            "p NUMERIC, "
            "z NUMERIC, "
            "T NUMERIC, "
            "Td NUMERIC, "
            "qvap NUMERIC, "
            "qliq NUMERIC"
        "); "
        "CREATE TABLE IF NOT EXISTS hatpro("
            "id INTEGER PRIMARY KEY, "
            "kind TEXT, "
            "valid NUMERIC, "
            "angle NUMERIC, "
            "TB_22240MHz NUMERIC, "
            "TB_23040MHz NUMERIC, "
            "TB_23840MHz NUMERIC, "
            "TB_25440MHz NUMERIC, "
            "TB_26240MHz NUMERIC, "
            "TB_27840MHz NUMERIC, "
            "TB_31400MHz NUMERIC, "
            "TB_51260MHz NUMERIC, "
            "TB_52280MHz NUMERIC, "
            "TB_53860MHz NUMERIC, "
            "TB_54940MHz NUMERIC, "
            "TB_56660MHz NUMERIC, "
            "TB_57300MHz NUMERIC, "
            "TB_58000MHz NUMERIC, "
            "p NUMERIC, "
            "T NUMERIC, "
            "qvap NUMERIC, "
            "rain INTEGER"
        "); "
        "CREATE TABLE IF NOT EXISTS nordkette("
            "valid NUMERIC PRIMARY KEY, "
            "T_710m NUMERIC,"
            "T_920m NUMERIC,"
            "T_1220m NUMERIC,"
            "T_2270m NUMERIC"
        "); "
        )


def select_files(files):
    """Select a subset of all files according to the specifications in args."""
    files = list(sorted(files))
    if args.offset is not None:
        files = files[int(args.offset):]
    if args.size is not None:
        files = files[:int(args.size)]
    return files


def filename(path):
    return path.split("/")[-1]
    # Yeah, yeah I should use os.pathlib... Sorry Windows-people :(


def read_raso_fwf(pid, path):
    """Read a fixed-width format radiosonde file.
    
    These are the ones containing the climatology that was also used by
    Giovanni Massaro and Daniel Meyer.
    """
    colspecs = [(8, 17), (17, 26), (26, 36), (43, 49)]
    names = ["p", "z", "T", "Td"]
    def errfloat(x):
        return None if "/" in x else float(x)
    file = filename(path)
    valid = (dt.datetime
            .strptime(file, "%Y%m%d_%H%M.reduced.txt")
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
            )
    df = pd.read_fwf(path, colspecs=colspecs, names=names,
            converters={n: errfloat for n in names},
            skiprows=1)
    df["T"] = 273.15 + df["T"]
    df["Td"] = 273.15 + df["Td"]
    ps = pd.Series(np.repeat(pid, len(df)), name="profile")
    # Calculate specific humidity and cloud water content
    qvap = fml.qvap(df["p"], df["Td"])
    qliq = fml.qliq(df["z"], df["p"], df["T"], df["Td"])
    data = pd.concat([ps, df, qvap, qliq], axis=1).as_matrix().tolist()
    cloudy = 1 if (qliq > 0).any() else 0
    return pid, data, valid, cloudy, file


def read_l2e(pid, path):
    """COSMO7 simulated soundings come as l2e files which are a sequence of
    headers and fixed-width format tables."""
    from l2e import parse as parse_l2e
    with open(path, "r") as f:
        for valid, run, df in parse_l2e(f, wmonrs=["11120"]):
            valid = valid.timestamp()
            run = run.timestamp()
            lead = valid - run
            # Clouds in COSMO7 output are always at 100 % RH
            qliq = fml.qliq(df["T"], df["qcloud"])
            ps = pd.Series(np.repeat(pid, len(df)), name="profile")
            data = pd.concat([ps, df[["p", "z", "T", "Td", "qvap"]], qliq],
                    axis=1).dropna(axis=0).as_matrix().tolist()
            cloudy = 1 if (qliq > 0).any() else 0
            yield pid, data, valid, valid-run, cloudy, filename(path)
            pid = pid + 1


def read_bufr_group(pid, paths):
    """Additional radiosonde profiles are available form the ertel2 archive
    in the form of BUFR files.
    
    /!\ BUFRReader is very (!) slow.
    """
    from bufr import BUFRReader
    reader = BUFRReader()
    for _, path in sorted(paths, key=lambda p: -os.path.getsize(p[1])):
        dfs, metadata = reader.read(path)
        try: df = dfs[-1][["p", "z", "T", "Td"]]
        except KeyError: continue
        valid = metadata["datetime"].timestamp()
        ps = pd.Series(np.repeat(pid, len(df)), name="profile")
        # Calculate specific humidity and cloud water content
        qvap = fml.qvap(df["p"], df["Td"])
        qliq = fml.qliq(df["z"], df["p"], df["T"], df["Td"])
        data = pd.concat([ps, df, qvap, qliq], axis=1).as_matrix().tolist()
        cloudy = 1 if (qliq > 0).any() else 0
        if cloudy and metadata["cloud cover"] == 0:
            print("Warning: {} | No clouds reported but found by RH criterion.".format(filename(path)))
        return pid, data, valid, cloudy, filename(path)


def read_netcdf(file):
    """Simulated HATPRO data are available from yearly netcdf files compatible
    with the IDL scripts that Giovanni Massaro and Daniel Meyer used."""
    import xarray
    xdata = xarray.open_dataset(file)
    substitutions = [
            ["n_angle", "elevation_angle"],
            ["n_date", "date"],
            ["n_frequency", "frequency"],
            ["n_height", "height_grid"]
            ]

    for origin, target in substitutions:
        xdata.coords[origin] = xdata.data_vars[target]
    def to_date(d):
        return (dt.datetime
                .strptime(str(d), "%Y%m%d%H")
                .replace(tzinfo=dt.timezone.utc)
                .timestamp()
                )
    df = xdata["brightness_temperatures"].to_dataframe()
    df = df.ix[~df.index.duplicated(keep="first")]
    ddf = df.unstack(level=[1,2,3])
    ddf.columns = ["{:>4}_f{}".format(round(c[2], 1), round(c[3]*1000)) for c in ddf.columns]
    angles = set([c.split("_")[0] for c in ddf.columns])
    for a in sorted(angles):
        data = ddf[list(sorted(c for c in ddf.columns if c.startswith(a)))]
        psfc = xdata["atmosphere_pressure_sfc"].to_dataframe()["atmosphere_pressure_sfc"]/100
        psfc.name = "p"
        Tsfc = xdata["atmosphere_temperature_sfc"].to_dataframe()["atmosphere_temperature_sfc"]
        Tsfc.name = "T"
        qsfc = xdata["atmosphere_humidity_sfc"].to_dataframe()["atmosphere_humidity_sfc"] / fml.ρ(p=psfc, T=Tsfc, e=0) # approx
        qsfc.name = "q"
        valid = pd.Series(data.index.map(to_date),
                index=data.index, name="valid")
        kind = pd.Series(np.repeat("igmk", [len(data)]),
                index=data.index, name="kind")
        angle = pd.Series(np.repeat(90-float(a), [len(data)]),
                index=data.index, name="angle")
        precip = pd.Series(np.repeat(None, [len(data)]),
                index=data.index, name="rain")
        data = pd.concat([kind, valid, angle, data, psfc,
                Tsfc, qsfc, precip], axis=1)
        yield data


parser = argparse.ArgumentParser()
parser.add_argument("--size", default=None)
parser.add_argument("--offset", default=None)
parser.add_argument("--pid", default=1)
parser.add_argument("--create", action="store_true")
parser.add_argument("data", nargs="?", default="")
parser.add_argument("output")

if __name__ == "__main__":

    args = parser.parse_args()
    pid = int(args.pid)

    db = Database(args.output)

    if args.create:
        db.execute(scheme)


    if args.data == "raso_fwf":
        files = select_files(glob("../data/raso_fwf/*.reduced.txt"))
        print("{} | {} | reading {} files".format(
                args.data, args.output, len(files)))
        rows, profiles = [], []
        for path in files:
            pid, data, valid, cloudy, file = read_raso_fwf(pid, path)
            profiles.append([pid, "raso", valid, 0., cloudy, file])
            rows.extend(data)
            pid = pid + 1
        print("{} | {} | writing {} of {} profiles... ".format(
                args.data, args.output, len(profiles), len(files)), end="")
        # Data is inserted in a dedicated if-block below


    if args.data == "raso_cosmo7":
        files = select_files(glob("../data/raso_cosmo7/*.l2e"))
        print("{} | {} | reading {} files".format(
                args.data, args.output, len(files)))
        rows, profiles = [], []
        for path in files:
            for pid, data, valid, lead, cloudy, file in read_l2e(pid, path):
                profiles.append([pid, "cosmo7", valid, lead, cloudy, file])
                rows.extend(data)
            pid = pid + 1
        print("{} | {} | writing {} profiles... ".format(
                args.data, args.output, len(profiles)), end="")
        # Data is inserted in a dedicated if-block below


    elif args.data == "raso_bufr":
        def get_datestr(path):
            file = filename(path)
            return file[:8] + file[10:12] if "." in file[:10] else file[:10]
        files = [(get_datestr(f), f) for f in glob("../data/raso_bufr/*.bfr")]
        groups = groupby(itemgetter(0), (f for f in files if ("309052" in f[1]
                or "30905" not in f[1])))
        files = select_files(groups.values())
        print("{} | {} | reading {} files".format(
                args.data, args.output, len(files)))
        rows, profiles = [], []
        for group in files:
            res = read_bufr_group(pid, group)
            if res is None: continue
            pid, data, valid, cloudy, file = res
            profiles.append([pid, "raso", valid, 0., cloudy, file])
            rows.extend(data)
            pid = pid + 1
        print("{} | {} | writing {} profiles... ".format(
                args.data, args.output, len(profiles)), end="")
        # Data is inserted in a dedicated if-block below


    if args.data.startswith("raso"):
        query = ("INSERT INTO profiles (id, kind, valid, lead, cloudy, file) "
                "VALUES (?, ?, ?, ?, ?, ?);")
        db.executemany(query, profiles)
        query = ("INSERT INTO profiledata (profile, p, z, T, Td, qvap, qliq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?);")
        db.executemany(query, rows)
        print("done!")

    
    if args.data == "nordkette":
        files = [
                "Alpenzoo_710m",
                "Hungerburg_920m",
                "Rastlboden_1220m",
                "Hafelekar_2270m"
                ]
        print("nordkette | {} | reading {} files".format(
                args.output, len(files)))
        def to_date(d):
            return (dt.datetime
                    .strptime(d, "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=dt.timezone.utc)
                    .timestamp()
                    )
        def temperature(x):
            return (None if not -499 < float(x) < 500
                    else round(float(x)/10 + 273.15, 2))
        data = []
        for station in files:
            df = pd.read_fwf("../data/stations/TEMPIS_{}.txt".format(station),
                    colspecs=[(2, 21), (22, 30)],
                    names=["valid", station],
                    converters={"valid": to_date, station: temperature},
                    skiprows=[0]).set_index("valid")
            data.append(df[station])
        rows = pd.concat(data, axis=1).reset_index().as_matrix().tolist()
        print("nordkette | {} | writing {} measurements... ".format(
                args.output, len(data)), end="")
        query = ("INSERT INTO nordkette (valid, T_710m, T_920m, T_1220m, "
                "T_2270m) VALUES (?, ?, ?, ?, ?);")
        db.executemany(query, rows)
        print("done!")


    if args.data == "igmk":
        files = glob("../data/hatpro_netcdf/rt*.cdf")
        print("{} | {} | reading {} files".format(
                args.data, args.output, len(files)))
        rows = []
        for file in sorted(files):
            for df in read_netcdf(file):
                rows.extend(df.as_matrix().tolist())
        print("{} | {} | writing {} rows... ".format(
                args.data, args.output, len(rows)), end="")
        query = ("INSERT INTO hatpro (kind, valid, angle, "
                "TB_22240MHz, TB_23040MHz, TB_23840MHz, TB_25440MHz, "
                "TB_26240MHz, TB_27840MHz, TB_31400MHz, TB_51260MHz, "
                "TB_52280MHz, TB_53860MHz, TB_54940MHz, TB_56660MHz, "
                "TB_57300MHz, TB_58000MHz, p, T, qvap, rain) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?);")
        db.executemany(query, rows)
        print("done!")
    

    if args.data == "hatpro":
        import hatpro
        def get_df(files, kind=None):
            dfs = []
            for f in files:
                dfs.append(hatpro.read(f))
            out = pd.concat(dfs, axis=0).reset_index(drop=True)
            if kind is not None:
                kc = pd.Series(np.repeat(kind, len(out)),
                        index=out.index, name="kind")
                out = pd.concat([out, kc], axis=1)
            return out
        met_files = glob("../data/hatpro_raw/MET/*.MET")
        brt_files = glob("../data/hatpro_raw/BRT/*.BRT")
        blb_files = glob("../data/hatpro_raw/BLB/*.BLB")
        print("{} | {} | reading {}*3 files".format(
                args.data, args.output, len(met_files)))
        met_df = get_df(met_files)
        brt_df = get_df(brt_files, kind="hatpro_brt")
        blb_df = get_df(blb_files, kind="hatpro_blb")
        bt_df = (pd.concat([brt_df, blb_df], axis=0)
                .sort_values(by=["valid", "angle"])
                .set_index("valid")
                )
        met_df2 = (met_df.sort_values(by="valid")
                .set_index("valid")
                .reindex(bt_df.index, method="nearest", tolerance=60)
                )
        qvap = fml.qvap(p=met_df2["p"], RH=met_df2["RH"], T=met_df2["T"])
        met_df2 = (met_df2
                .drop("rain", 1)
                .drop("ff", 1)
                .drop("dd", 1)
                .drop("RH", 1)
                )
        df = pd.concat([bt_df, met_df2, qvap], axis=1).reset_index()
        rows = df.as_matrix().tolist()
        print("{} | {} | writing {} rows... ".format(
                args.data, args.output, len(rows)), end="")
        query = ("INSERT INTO hatpro (valid, rain, "
                "TB_22240MHz, TB_23040MHz, TB_23840MHz, TB_25440MHz, "
                "TB_26240MHz, TB_27840MHz, TB_31400MHz, TB_51260MHz, "
                "TB_52280MHz, TB_53860MHz, TB_54940MHz, TB_56660MHz, "
                "TB_57300MHz, TB_58000MHz, angle, kind, p, T, qvap) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?);")
        db.executemany(query, rows)
        print("done!")


    if args.data == "description":
        print("{} | {} | creating and filling description table... ".format(
                args.data, args.output), end="")
        db.execute("CREATE TABLE IF NOT EXISTS description(tablename TEXT, "
                "kind TEXT, text TEXT);")
        rows = [["profiles", "cosmo7",
                        "COSMO7 simulated soundings for Innsbruck"],
                ["profiles", "raso",
                        "Radiosonde profiles from Innsbruck airport"],
                ["hatpro", "igmk",
                        "Simulations of HATPRO measurements based on "
                        "hires radiosonde profiles done by IGMK"],
                ["hatpro", "hatpro_blb",
                        "HATPRO boundary layer elevation scan measurements "
                        "from the ACINN roof"],
                ["hatpro", "hatpro_brt",
                        "HATPRO hifreq measurements from the ACINN roof"],
                ["nordkette", None,
                        "Nordkette slope temperature measurements by ZAMG"]
                ]
        query = "INSERT INTO description (tablename, kind, text) VALUES (?, ?, ?);"
        db.executemany(query, rows)
        print("done!")

