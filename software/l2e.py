"""MeteoSwiss L2E file format parser."""

from itertools import islice
from io import StringIO
import datetime as dt

import pandas as pd


__all__ = ["parse"]


def advancen(f, n):
    for _ in range(n): next(f)

def readn(f, n):
    return "".join(f.readline() for _ in range(n))

def read_header(f):
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()
    if (not line1.startswith("H") or not line2.startswith("M")
            or not line3.startswith("*")):
        raise ValueError("File is not at a header position.")
    #      valid        run           wmo nr      levels
    return line1[3:11], line3[24:32], line3[3:8], line3[34:37]

def decifloat(x):
    return float(x)/10.

def decifloatkelvin(x):
    return float(x)/10. + 273.15

def read_profile(f, n):
    colspecs = [(2, 7), (7, 12), (14, 18), (20, 24), (38, 48), (49, 59)]
    names = ["p", "z", "T", "Td", "qvap", "qcloud"]
    converters = {"p": decifloat, "z": float, "T": decifloatkelvin,
            "Td": decifloat, "qvap": float, "qcloud": float}
    df = pd.read_fwf(StringIO(readn(f, n)), colspecs=colspecs, names=names,
            converters=converters, na_values=["-.9999E+04"])
    df["Td"] = df["T"] - df["Td"]
    return df

def parse_time(d):
    return dt.datetime.strptime(d, "%y%m%d%H").replace(tzinfo=dt.timezone.utc)

def parse(f, wmonrs=None):
    while True:
        try: valid, run, wmonr, levels = read_header(f)
        except ValueError:
            break
        levels = int(levels)
        if wmonrs is None or wmonr in wmonrs:
            data = read_profile(f, levels-2)
            assert f.readline().startswith("25-9-9-9-9-9")
            valid = parse_time(valid)
            run = parse_time(run)
            yield valid, run, data
        else:
            advancen(f, levels-1)

