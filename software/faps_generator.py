"""Train FAPs for HATPRO channels and generate code."""

from functools import partial

import mwrt
from mwrt.fapgen import (
        CloudFAPGenerator, GasFAPGenerator, as_absorption, generate_code
        )


νs = ["22240", "23040", "23840", "25440", "26240", "27840", "31400",
      "51260", "52280", "53860", "54940", "56660", "57300", "58000"]

degrees = [3]*7 + [3]*7

clw_absorp = as_absorption(mwrt.tkc.refractivity_lwc)
gas_absorp = as_absorption(mwrt.liebe93.refractivity_gaseous)

if __name__ == "__main__":

    first = True
    out = []
    fapnames, tbnames = [], []
    for ν, deg in zip(νs, degrees):
        print("FAP for " + ν + " MHz")
        νi = int(ν)/1000
        # Fit a Cloud FAP for the selected frequency
        cfap = CloudFAPGenerator(degree=5)
        cfap.train_T_range = 233., 303., 500000
        print("    fitting cloud absorption...")
        cfap.fit(partial(clw_absorp, νi))
        # Fit a Gas FAP for the selected frequency
        gfap = GasFAPGenerator(degree=deg, interaction=False, alpha=10.)
        gfap.train_p_range = 115., 980., 100
        gfap.train_T_range = 170., 313., 100
        gfap.train_rh_range = 0., 1., 120
        print("    fitting gas absorption...")
        gfap.fit(partial(gas_absorp, νi))
        # Store FAP code
        fapnames.append("FAP" + ν + "MHz")
        tbnames.append("TB_" + ν + "MHz")
        print("    generating code...")
        out.append(generate_code(fapnames[-1], gfap, cfap, with_import=first))
        first = False
    out.append("faps =  [" + ", ".join(fapnames) + "]\n")
    out.append("tbs =  [\"" + "\", \"".join(tbnames) + "\"]\n")

    print("Writing to file....")
    with open("faps_hatpro.py", "w") as f:
        f.write("\n".join(out))
    print("All done.")

