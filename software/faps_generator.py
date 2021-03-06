"""Train FAPs for HATPRO channels and generate code."""

from functools import partial

import mwrt
from mwrt.background import USStandardBackground
from mwrt.fapgen import (
        CloudFAPGenerator, GasFAPGenerator, as_absorption, generate_code
        )


# Considered frequencies in MHz
νs = ["22240", "23040", "23840", "25440", "26240", "27840", "31400",
      "51260", "52280", "53860", "54940", "56660", "57300", "58000"]

# Polynomial degrees and interaction term degrees for each frequency
degrees =  [3, 3, 3, 3, 3, 3, 3,  3, 3, 3, 3, 3, 3, 3]
idegrees = [3, 3, 3, 3, 3, 3, 3,  2, 2, 2, 2, 2, 2, 2]
# Tests showed that V band frequencies do not need as many interaction terms
# as K band to be similarly accurate

# Absorption models the FAPs are trained to emulate
clw_absorp = as_absorption(mwrt.tkc.refractivity_lwc)
gas_absorp = as_absorption(mwrt.liebe93.refractivity_gaseous)


if __name__ == "__main__":

    # Also calculate cosmic background temperatures adjusted to compensate
    # stratospheric emission in the selected channels. Set up US Standard
    # Atmosphere temperature profile:
    usatm = USStandardBackground(lower=12612, upper=40000)

    first = True
    out = []
    bgs, freqs = [], []
    fapnames, tbnames = [], []
    for ν, deg, ideg in zip(νs, degrees, idegrees):
        print("FAP for " + ν + " MHz")
        νi = int(ν)/1000
        freqs.append("{:5.2f}".format(νi))
        # Fit a Cloud FAP for the selected frequency
        cfap = CloudFAPGenerator(degree=5, alpha=1.)
        cfap.train_T_range = 233., 303., 500000
        print("    cloud absorption...")
        cfap.fit(partial(clw_absorp, νi))
        # Fit a Gas FAP for the selected frequency
        gfap = GasFAPGenerator(degree=deg, interaction_degree=ideg, alpha=100.)
        gfap.train_p_range = 150., 980., 100
        gfap.train_T_range = 170., 313., 100
        gfap.train_rh_range = 0., 1., 120
        print("    gas absorption...")
        absorp = partial(gas_absorp, νi)
        gfap.fit(absorp)
        # Store FAP code
        fapnames.append("FAP" + ν + "MHz")
        tbnames.append("TB_" + ν + "MHz")
        out.append(generate_code(fapnames[-1], gfap, cfap, with_import=first))
        # Adjusted cosmic background TB
        print("    background TB...")
        bgs.append(usatm.evaluate(absorp, angle=0., cosmic=2.75))
        first = False
    out.append("freqs = [" + ", ".join(freqs) + "]\n")
    out.append("bgs = [" + ", ".join(str(round(x, 5)) for x in bgs) + "]\n")
    out.append("faps = [" + ", ".join(fapnames) + "]\n")
    out.append("tbs = [\"" + "\", \"".join(tbnames) + "\"]\n")

    print("Writing to file....")
    with open("faps_hatpro.py", "w") as f:
        f.write("\n".join(out))
    print("done.")

