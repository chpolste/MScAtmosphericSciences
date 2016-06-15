import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", default=None)
parser.add_argument("--offset", default=None)
parser.add_argument("--pid", default=1)
parser.add_argument("data")
parser.add_argument("output")


def select_files(files):
    files = list(files)
    if args.offset is not None:
        files = files[int(args.offset):]
    if args.size is not None:
        files = files[:int(args.size)]
    return files

def errfloat(x):
    return None if "/" in x else float(x)

if __name__ == "__main__":

    args = parser.parse_args()

    import json, os
    from glob import glob
    import datetime as dt

    import numpy as np
    import pandas as pd

    from dbtoolbox import DatabaseToolbox

    db = DatabaseToolbox(args.output)
    db.create_tables()

    pid = int(args.pid)
    
    # Radiosondes in fixed width text format
    if args.data == "raso_fwf":
        files = select_files(sorted(glob("../data/raso_ibk_fwf/*.reduced.txt")))
        colspecs = [(8, 17), (17, 26), (26, 36), (43, 49)]
        names = ["p", "geopot", "T", "Td"]
        converters = {"p": errfloat, "geopot": errfloat, "T": errfloat, "Td": errfloat}
        print("raso_fwf | importing {} files to {}".format(len(files), args.output))
        rowcache, profilecache = [], []
        for file in files:
            filename = file.split("/")[-1]
            valid = (dt.datetime
                     .strptime(filename, "%Y%m%d_%H%M.reduced.txt")
                     .replace(tzinfo=dt.timezone.utc)
                     .timestamp()
                     )
            metadata = {"file": filename}
            profilecache.append([pid, valid, json.dumps(metadata)])
            df = pd.read_fwf(file, colspecs=colspecs, names=names, converters=converters, skiprows=1)
            ps = pd.Series(np.repeat(pid, len(df)), name="parent")
            rowcache.extend(pd.concat([ps, df], axis=1).as_matrix().tolist())
            pid = pid + 1
        print("raso_fwf | writing {} of {} profiles to database... ".format(len(profilecache), len(files)), end="")
        query = "INSERT INTO raso (id, valid, metadata) VALUES (?, ?, ?);"
        db.executemany(query, profilecache)
        query = "INSERT INTO rasodata (parent, p, geopot, T, Td) VALUES (?, ?, ?, ?, ?);"
        db.executemany(query, rowcache)
        print("done!")


    # Radiosondes in bufr format
    elif args.data == "raso_bufr":
        from collections import namedtuple
        from operator import itemgetter
        from toolz import groupby
        from bufr import BUFRReader

        reader = BUFRReader()
        fs = glob("../data/raso_ibk_bufr/*.bfr")
        bufrfile = namedtuple("bufrfile", ["date", "file"])
        get_datestr = lambda file: file[:8] + file[10:12] if "." in file[:10] else file[:10]
        fs = [bufrfile(get_datestr(f.split("/")[-1]), f) for f in fs]
        raso52 = groupby(itemgetter(0), filter(lambda f: "309052" in f[1], fs))
        nones = groupby(itemgetter(0), filter(lambda f: "30905" not in f[1], fs))
        for key, val in nones.items():
            if key in raso52: raso52[key].extend(val)
            else: raso52[key] = val

        def select_largest(sondes):
            out = sondes[0]
            for s in sondes:
                if os.path.getsize(s.file) > os.path.getsize(out.file):
                    out = s
            return out
        files = [select_largest(sondes).file for sondes in raso52.values()]
        files = select_files(sorted(files))
        print("raso_bufr | importing {} files to {}".format(len(files), args.output))
        valids, rowcache, profilecache = set(), [], []
        for file in files:
            filename = file.split("/")[-1]
            df, metadata = reader.read(file)
            try: df = df[["p", "geopot", "T", "Td"]]
            except KeyError: continue
            valid = metadata["datetime"].timestamp()
            if valid in valids: continue
            valids.add(valid)
            metadata = {"file": filename, "cloud_cover": metadata["cloud cover"]}
            profilecache.append([pid, valid, json.dumps(metadata)])
            ps = pd.Series(np.repeat(pid, len(df)), name="parent")
            rowcache.extend(pd.concat([ps, df], axis=1).as_matrix().tolist())
            pid = pid + 1
        
        print("raso_bufr | writing {} of {} profiles to database... ".format(len(profilecache), len(files)), end="")
        query = "INSERT INTO raso (id, valid, metadata) VALUES (?, ?, ?);"
        db.executemany(query, profilecache)
        query = "INSERT INTO rasodata (parent, p, geopot, T, Td) VALUES (?, ?, ?, ?, ?);"
        db.executemany(query, rowcache)
        print("done!")
