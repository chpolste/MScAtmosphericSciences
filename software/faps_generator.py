"""Train FAPs for HATPRO channels and generate code."""

from functools import partial

import mwrt
from mwrt.fapgen import (
        CloudFAPGenerator, GasFAPGenerator, as_absorption, generate_code
        )


νs = ["22240", "23040", "23840", "25440", "26240", "27840", "31400",
      "51260", "52280", "53860", "54940", "56660", "57300", "58000"]

# The FAPs in the K-Band seem to perform quite badly at low pressures with
# a polynomial of degree 3. They are therefore fitted with degree 4.
degrees = [4, 4, 4, 4, 4, 4, 4,  3, 3, 3, 3, 3, 3, 3]


clw_absorp = as_absorption(mwrt.tkc.refractivity_lwc)
gas_absorp = as_absorption(mwrt.liebe93.refractivity_gaseous)


if __name__ == "__main__":

    first = True
    out = []
    for ν, deg in zip(νs, degrees):
        print("FAP for " + ν + " MHz")
        νi = int(ν)/1000
        # Fit a Cloud FAP for the selected frequency
        cfap = CloudFAPGenerator(degree=5)
        print("    fitting cloud absorption...")
        cfap.fit(partial(clw_absorp, νi))
        # Fit a Gas FAP for the selected frequency
        gfap = GasFAPGenerator(degree=deg)
        print("    fitting gas absorption...")
        gfap.fit(partial(gas_absorp, νi))
        # Store FAP code
        name = "FAP" + ν + "MHz"
        print("    generating code...")
        out.append(generate_code(name, gfap, cfap, with_import=first))
        first = False

    print("Writing to file....")
    with open("faps_hatpro.py", "w") as f:
        f.write("\n".join(out))
    print("All done.")

