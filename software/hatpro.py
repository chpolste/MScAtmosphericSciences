"""HATPRO raw data readers.

Implemented: BRT, BLB, MET. All readers return pandas dataframes.

Based on Reto Stauffer's reader classes, but functional and using memmap
instead of structs and loops for the data section.
"""

import struct
import datetime as dt

import numpy as np
import pandas as pd


__all__ = ["read"]


def _read_struct(file, types):
    size = struct.calcsize(types)
    out = struct.unpack("<"+types, file.read(size))
    return out[0] if len(out) == 1 else out

def _data_to_df(path, offset, dtype):
    memmap = np.memmap(path, dtype=dtype, mode="r", offset=offset, order="C")
    return pd.DataFrame(memmap)

def _to_utc(valid):
    """Get UTC timestamp from HATPRO time format (seconds since 2001-01-01."""
    return valid + dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc).timestamp()

def _int_rain(rain):
    """Turn character rainflag into a 0/1 integer flag."""
    return (rain != b"").astype(int)


def read(path):
    """Generic HATPRO file reader. Dispatches to the appropriate function."""
    with open(path, "rb") as file:
        code = _read_struct(file, "I")
        file.seek(0) # Rewind
        if code == 599658944: return read_MET(file)
        if code == 666666: return read_BRT(file)
        if code == 567845848: return read_BLB(file)
        raise NotImplementedError("No reader for file format available.")


def read_MET(file):
    """Load a HATPRO MET file as a dataframe."""
    code, n_rec, additional_sensors = _read_struct(file, "IIc")
    p_min, p_max, T_min, T_max, H_min, H_max = _read_struct(file, "ffffff")
    sensors = ["p", "T", "RH"]
    if ord(additional_sensors) in [1, 3]:
        sensors.append("ff")
        ff_min, ff_max = _read_struct(file, "ff")
    if ord(additional_sensors) in [2, 3]:
        sensors.append("dd")
        dd_min, dd_max = _read_struct(file, "ff")
    timeref = _read_struct(file, "I")
    assert code == 599658944 and timeref == 1 # UTC
    dtype = np.dtype(
            [("valid", np.int32), ("rain", "S1")]
            + [(s, np.float32) for s in sensors]
            )
    df = _data_to_df(file.name, file.tell(), dtype)
    assert(len(df) == n_rec)
    df["valid"] = _to_utc(df["valid"])
    df["rain"] = _int_rain(df["rain"])
    return df


def read_BRT(file):
    """Load a HATPRO BRT file as a dataframe."""
    code, n_rec, timeref, n_freq = _read_struct(file, "IIII")
    assert code == 666666 and timeref == 1 # UTC
    freqname = "TB_{:<5.0f}MHz".format
    freqs = [freqname(x*1000) for x in _read_struct(file, "f"*n_freq)]
    TB_min = _read_struct(file, "f"*n_freq)
    TB_max = _read_struct(file, "f"*n_freq)
    dtype = np.dtype(
            [("valid", np.int32), ("rain", "S1")]
            + [(f, np.float32) for f in freqs]
            + [("angle", np.float32)]
            )
    df = _data_to_df(file.name, file.tell(), dtype)
    assert(len(df) == n_rec)
    df["valid"] = _to_utc(df["valid"])
    df["rain"] = _int_rain(df["rain"])
    df["angle"] = 90 - df["angle"] # Elevation to zenith angle
    return df


def read_BLB(file):
    """Load a HATPRO BLB file as a dataframe."""
    # Read header
    code, n_rec, n_freq = _read_struct(file, "III")
    BLB_min = _read_struct(file, "f"*n_freq)
    BLB_max = _read_struct(file, "f"*n_freq)
    timeref = _read_struct(file, "I")
    assert code == 567845848 and timeref == 1 # UTC
    freqname = "TB_{:<5.0f}MHz".format
    freqs = [freqname(x*1000) for x in _read_struct(file, "f"*n_freq)]
    n_ang = _read_struct(file, "I")
    angles = [90-round(x, 2) for x in _read_struct(file, "f"*n_ang)]
    dtype = [("valid", np.int32), ("rain", "S1")]
    for f in freqs:
        dtype.extend([(f+"_"+str(a), np.float32) for a in angles])
        dtype.append(("T_"+f, np.float32))
    df = _data_to_df(file.name, file.tell(), np.dtype(dtype))
    assert(len(df) == n_rec)
    df["rain"] = _int_rain(df["rain"])
    # Reduce columns in dataframe by making the angle an explicit column
    # (maybe this can be done better with DataFrame.stack?)
    out = [_select_angle(df, a) for a in angles]
    return pd.concat(out, axis=0).reset_index(drop=True)


def _select_angle(df, angle):
    """Helper function for column number reduction in read_BLB."""
    cols = ["valid", "rain"]
    outcols = ["valid", "rain"]
    suffix = "_"+str(angle)
    for col in df.columns:
        if col.endswith(suffix):
            cols.append(col)
            outcols.append(col[:-len(suffix)])
    angles = pd.Series(np.repeat(angle, len(df)), index=df.index)
    outcols.append("angle")
    out = pd.concat([df[cols], angles], axis=1)
    out.columns = outcols
    return out

