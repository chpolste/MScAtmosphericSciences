"""Optimal estimation retrievals for actual brightness temperature data."""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from db_tools import (read_csv_profiles, read_csv_covariance,
        iter_profiles, split_bands, get_zenith, read_csv_mean)
from optimal_estimation import (VirtualHATPRO, VirtualHATPRO_zenith,
        VirtualHATPRO_Vband, Gaussian, rgrid, mgrid, z_hatpro, z_top,
        iterate_to_convergence)

parser = argparse.ArgumentParser()
parser.add_argument("kind", type=str, help="""
        Where to get prior from? raso (= COSMO-7, this is mislabled), continuous
        (= previous retrieval with COSMO-7 uncertainty)
        """)
parser.add_argument("bias", type=str, help="""
        Should the instrumental bias be corrected? yes/no
        """)

if __name__ == "__main__":

    args = parser.parse_args()

    # The observation covariance matrix is based on MWRTM error covariance
    # calculated from a comparison with IGMK data
    obs_cov = read_csv_covariance("../data/unified/priors/TB_mwrtm_fap_igmk_cov.csv")
    # Add 0.5 K uncorrelated instrument noise (0.5 K std = 0.25 K² cov)
    obs_cov = obs_cov + 0.25*np.eye(obs_cov.shape[0])
    obs_error = Gaussian(np.zeros(obs_cov.shape[0]), obs_cov) # bias corrected in ys

    # COSMO-7 priors
    cosmo_cov = read_csv_covariance("../data/unified/priors/x_cosmo7+00+06_cov.csv")
    cosmo_means = read_csv_profiles("../data/unified/priors/x_cosmo7+00+06_mean.csv")

    # Raso data for initial values of continuous retrieval
    Traso = read_csv_profiles("../data/unified/test/T_rasoclim.csv")
    lnqraso = read_csv_profiles("../data/unified/test/lnq_rasoclim.csv")
    xraso = pd.concat([Traso.add_prefix("T_"), lnqraso.add_prefix("lnq_")], axis=1)

    p = read_csv_profiles("../data/unified/test/sfc_hatpro.csv")["p"]
    ys = read_csv_profiles("../data/unified/test/TB_hatpro.csv")
    
    name = "_optest"

    if args.bias == "yes":
        name += "_biased"
    elif args.bias == "no":
        # Bias has to be subtracted from model therefore added to HATPRO
        ys = ys + read_csv_mean("../data/unified/priors/TB_mwrtm_fap_bias.csv")
    else: raise ValueError()

    vh = VirtualHATPRO(z_retrieval=rgrid, z_model=mgrid, error=obs_error)

    if args.kind == "continuous":
        # Selected period of investigation:
        ys = ys.loc["2015-10-28 02:15:05":"2015-10-29 02:15:06",:]
        name += "_continuous"
    elif args.kind == "raso":
        ys = ys.reindex(cosmo_means.index, method="nearest", tolerance=dt.timedelta(minutes=30)).dropna()
        p = p.reindex(cosmo_means.index, method="nearest", tolerance=dt.timedelta(minutes=30)).dropna()
        name += "_raso"
    else: raise ValueError()

    i = 1
    valids, convergeds, iterations, states = [], [], [], []
    
    start = xraso["2015-10-28 02:15:05"]
    # TODO: starting with this date is ok for continuous retrieval but produces
    #       an additional row of data for retrievals with COSMO-7 that is
    #       unwanted. This is a bug.
    convergeds.append(True)
    iterations.append(0)
    states.append(start.values.flatten())
    valids.append(start.index[0])
    
    for valid in ys.index:
        p0 = float(p[valid])
        y = ys.loc[valid,:].values.reshape(-1, 1)
        if args.kind == "continuous":
            prior = Gaussian(states[-1].reshape(-1, 1), cosmo_cov)
        else:
            prior = Gaussian(cosmo_means.loc[valid,:].values.reshape(-1, 1), cosmo_cov)

        ret = vh.retrieve(y, p0, prior.mean, prior, iterations=0)
        converged, best = iterate_to_convergence(ret)

        valids.append(valid)
        convergeds.append(converged)
        iterations.append(best)
        states.append(ret.μs[best].flatten())

        i += 1

        #if i > 3: break

    # TODO: again, this should only happen for the continuous retrievals.
    end = xraso["2015-10-28 02:15:05"]
    convergeds.append(True)
    iterations.append(0)
    states.append(end.values.flatten())
    valids.append(end.index[0])

    valids = pd.Series(valids, name="valid")
    pd.DataFrame(np.vstack([convergeds, iterations]).T, columns=["converged", "iterations"],
            index=valids).to_csv("../data/unified/retrievals/convergence" + name + ".csv")
    pd.DataFrame(np.vstack(states), columns=cosmo_means.columns, index=valids
            ).to_csv("../data/unified/retrievals/x" + name + ".csv")

