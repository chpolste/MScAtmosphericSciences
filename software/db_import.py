"""Process data from a given source and put it into a sqlite3 database.
    
This is used to parallelize the data processing as the used bufr-reader is
quite slow and would otherwise take over 2 hours to process all available
files. Instead of handling the parallelization in Python, it is done by
creating multiple databases in parallel with a bash script (db_create) and then
merging those databases. To maintain correct relations in the database the
profile-ids have to be offset for each database so that every id is unique
in the merged database. This is what the pid argument is for, it gives the
first profile index to be used. The size and offset arguments specify which
files are selected from the respective folders.
"""

import argparse, json, os
import datetime as dt
from glob import glob
from operator import itemgetter

from toolz import groupby
import numpy as np
import pandas as pd

from dbtoolbox import DatabaseToolbox
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
            "profile INTEGER, "
            "angle NUMERIC, "
            "f22240 NUMERIC, "
            "f23040 NUMERIC, "
            "f23840 NUMERIC, "
            "f25440 NUMERIC, "
            "f26240 NUMERIC, "
            "f27840 NUMERIC, "
            "f31400 NUMERIC, "
            "f51260 NUMERIC, "
            "f52280 NUMERIC, "
            "f53860 NUMERIC, "
            "f54940 NUMERIC, "
            "f56660 NUMERIC, "
            "f57300 NUMERIC, "
            "f58000 NUMERIC, "
            "p NUMERIC, "
            "z NUMERIC, "
            "T NUMERIC, "
            "Td NUMERIC, "
            "qvap NUMERIC, "
            "precip INTEGER "
        ");"
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
    qvap = fml.qvap(df["p"], df["Td"])
    qliq = fml.qliq(df["z"], df["p"], df["T"], df["Td"])
    qliq = pd.Series(qliq, name="qliq")
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
                    axis=1).as_matrix().tolist()
            cloudy = 1 if (qliq > 0).any() else 0
            yield pid, data, valid, valid-run, cloudy, filename(path)
            pid = pid + 1


def read_bufr_group(pid, paths):
    from bufr import BUFRReader
    reader = BUFRReader()
    for _, path in sorted(paths, key=lambda p: -os.path.getsize(p[1])):
        df, metadata = reader.read(path)
        try: df = df[["p", "z", "T", "Td"]]
        except KeyError: continue
        valid = metadata["datetime"].timestamp()
        ps = pd.Series(np.repeat(pid, len(df)), name="profile")
        qvap = fml.qvap(df["p"], df["Td"])
        qliq = fml.qliq(df["z"], df["p"], df["T"], df["Td"])
        qliq = pd.Series(qliq, name="qliq")
        data = pd.concat([ps, df, qvap, qliq], axis=1).as_matrix().tolist()
        cloudy = 1 if (qliq > 0).any() else 0
        if cloudy and metadata["cloud cover"] == 0:
            print("Warning: {} | No clouds reported but found by RH criterion.")
        return pid, data, valid, cloudy, filename(path)


parser = argparse.ArgumentParser()
parser.add_argument("--size", default=None)
parser.add_argument("--offset", default=None)
parser.add_argument("--pid", default=1)
parser.add_argument("data")
parser.add_argument("output")

if __name__ == "__main__":

    args = parser.parse_args()
    pid = int(args.pid)

    db = DatabaseToolbox(args.output)

    if args.data == "create_tables":
        db.execute(scheme)


    if args.data == "raso_fwf":
        files = select_files(glob("../data/raso_fwf/*.reduced.txt"))
        print("raso_fwf | {} | reading {} files".format(
                args.output, len(files)))
        rows, profiles = [], []
        for path in files:
            pid, data, valid, cloudy, file = read_raso_fwf(pid, path)
            profiles.append([pid, "raso", valid, 0., cloudy, file])
            rows.extend(data)
            pid = pid + 1
        print("raso_fwf | {} | writing {} of {} profiles... ".format(
                args.output, len(profiles), len(files)), end="")
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
        print("cosmo7_le2 | {} | writing {} profiles... ".format(
                args.output, len(profiles)), end="")
        # Data is inserted in a dedicated if-block below


    elif args.data == "raso_bufr":
        def get_datestr(path):
            file = filename(path)
            return file[:8] + file[10:12] if "." in file[:10] else file[:10]
        files = [(get_datestr(f), f) for f in glob("../data/raso_bufr/*.bfr")]
        groups = groupby(itemgetter(0), (f for f in files if ("309052" in f[1]
                or "30905" not in f[1])))
        files = select_files(groups.values())
        print("raso_bufr | {} | reading {} files".format(
                args.output, len(files)))
        rows, profiles = [], []
        for group in files:
            res = read_bufr_group(pid, group)
            if res is None: continue
            pid, data, valid, cloudy, file = res
            profiles.append([pid, "raso", valid, 0., cloudy, file])
            rows.extend(data)
            pid = pid + 1
        print("raso_bufr | {} | writing {} profiles... ".format(
                args.output, len(profiles)), end="")
        # Data is inserted in a dedicated if-block below


    if args.data.startswith("raso"):
        query = ("INSERT INTO profiles (id, kind, valid, lead, cloudy, file) "
                "VALUES (?, ?, ?, ?, ?, ?);")
        db.executemany(query, profiles)
        query = ("INSERT INTO profiledata (profile, p, z, T, Td, qvap, qliq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?);")
        db.executemany(query, rows)
        print("done!")


