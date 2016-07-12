"""Condenses relevant data from the database into a few csv files."""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from db_tools import Database
import formulas as fml


z_hatpro = 612
ret_grid = np.logspace(np.log10(z_hatpro), np.log10(15000), 50).astype(int)


def stretch(xs, lower=None, upper=None, power=1):
    n = len(xs)
    loffset = lower - xs[0] if lower is not None else 0.
    toffset = upper - xs[-1] if upper is not None else 0.
    out = xs + loffset * np.linspace(1, 0, n)**power
    out = out + toffset * np.linspace(0, 1, n)**power
    return out

def bt_dataset(data):
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


parser = argparse.ArgumentParser()
parser.add_argument("data", nargs="?", default="all")

if __name__ == "__main__":

    args = parser.parse_args()

    db = Database("../data/amalg.db")


    if args.data == "bt_raso" or args.data == "all":
        # IGMK processed brightness temperatures
        bt_igmk = igmk = db.as_dataframe("""
                SELECT * FROM hatpro
                WHERE kind = "igmk"
                ORDER BY valid ASC;
                """)
        # MWRTM brightness temperatures
        bt_mwrt = db.as_dataframe("""
                SELECT * FROM hatpro
                WHERE kind = "mwrt"
                ORDER BY valid ASC;
                """)
        bt_dataset(bt_igmk).to_csv("../data/unified/bt_igmk.csv")
        bt_dataset(bt_mwrt).to_csv("../data/unified/bt_mwrt.csv")


    if args.data == "cosmo7" or args.data == "all":
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
                    out[var].append(interp1d(zs, profile[var].values)(ret_grid))
            for var, values in out.items():
                (pd.DataFrame(np.vstack(values), index=valid,
                        columns=["T_{}m".format(int(z)) for z in ret_grid])
                        .sort_index()
                        .round(10)
                        .to_csv("../data/unified/{}_cosmo7+{:0>2}.csv".format(var, lead//3600))
                        )            
