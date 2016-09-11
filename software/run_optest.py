"""Optimal estimation retrievals for synthetic brightness temperature data."""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from db_tools import (read_csv_profiles, read_csv_covariance,
        iter_profiles, split_bands, get_zenith)
from optimal_estimation import (VirtualHATPRO, VirtualHATPRO_zenith,
        VirtualHATPRO_Vband, Gaussian, rgrid, mgrid, z_hatpro, z_top,
        iterate_to_convergence)


def get_prior(valid):
    """Get the prior distribution corresponding to a valid datetime."""
    return Gaussian(prior_means.loc[valid,:].values.reshape(-1, 1), prior_cov)


parser = argparse.ArgumentParser()
parser.add_argument("kind", type=str, help="""
        HATPRO configuration: full, vband, zenith
        """)
parser.add_argument("prior", type=str, help="""
        Prior distribution: cosmo, regression
        """)
parser.add_argument("guess", type=str, help="""
        First guess: cosmo, regression
        """)

if __name__ == "__main__":

    args = parser.parse_args()

    # The observation covariance matrix is based on MWRTM error covariance
    # calculated from a comparison with IGMK data
    obs_cov = read_csv_covariance("../data/unified/priors/TB_mwrtm_fap_igmk_cov.csv")
    _, obs_cov_v = split_bands(obs_cov)
    obs_cov_z = get_zenith(obs_cov)
    # Add 0.5 K uncorrelated instrument noise (0.5 K std = 0.25 K² cov)
    obs_cov = obs_cov + 0.25*np.eye(obs_cov.shape[0])
    obs_cov_v = obs_cov_v + 0.25*np.eye(obs_cov_v.shape[0])
    obs_cov_z = obs_cov_z + 0.25*np.eye(obs_cov_z.shape[0])
    # Synthetic data: no mean observation error (assumption)
    obs_error = Gaussian(np.zeros(obs_cov.shape[0]), obs_cov)
    obs_error_v = Gaussian(np.zeros(obs_cov_v.shape[0]), obs_cov_v)
    obs_error_z = Gaussian(np.zeros(obs_cov_z.shape[0]), obs_cov_z)

    p = read_csv_profiles("../data/unified/test/psfc.csv")

    ys = read_csv_profiles("../data/unified/test/TB_mwrtm.csv")
    ys = ys + np.random.normal(0., scale=0.5, size=ys.shape)
    _, ys_v = split_bands(ys)
    ys_z = get_zenith(ys)

    name = ""

    if args.prior == "cosmo":
        prior_cov = read_csv_covariance("../data/unified/priors/x_cosmo7+00+06_cov.csv")
        prior_means = read_csv_profiles("../data/unified/priors/x_cosmo7+00+06_mean.csv")
        name += "_cosmo"
    elif args.prior == "regression":
        prior_cov = read_csv_covariance("../data/unified/priors/x_regression_cov.csv")
        prior_means = read_csv_profiles("../data/unified/priors/x_regression_mean.csv")
        name += "_regression"
    else: raise ValueError()

    if args.guess == "cosmo":
        guesses = read_csv_profiles("../data/unified/priors/x_cosmo7+00+06_mean.csv")
        name += "_cosmo"
    elif args.guess == "regression":
        guesses = read_csv_profiles("../data/unified/priors/x_regression_mean.csv")
        name += "_regression"
    else: raise ValueError()

    if args.kind == "full":
        vh = VirtualHATPRO(z_retrieval=rgrid, z_model=mgrid, error=obs_error)
        obs = ys
        name += "_full"
    elif args.kind == "zenith":
        vh = VirtualHATPRO_zenith(z_retrieval=rgrid, z_model=mgrid, error=obs_error_z)
        obs = ys_z
        name += "_zenith"
    elif args.kind == "vband":
        vh = VirtualHATPRO_Vband(z_retrieval=rgrid, z_model=mgrid, error=obs_error_v)
        obs = ys_v
        name += "_vband"
    else: raise ValueError()

    # Virtual HATPRO is set up, data are loaded, start retrievals for each
    # available valid time
    i = 1
    valids, convergeds, iterations, states = [], [], [], []
    for valid in prior_means.index:
        #print("{} of {} ({:4.1f} %)".format(i, len(prior_means.index), i/len(prior_means.index)*100))
        try:
            prior = get_prior(valid)
            p0 = float(p.loc[valid,"p"])
            y = obs.loc[valid,:].values.reshape(-1, 1)
            guess = guesses.loc[valid,:].values.reshape(-1, 1)
        except KeyError:
            i += 1
            continue

        ret = vh.retrieve(y, p0, prior.mean, prior, iterations=0)
        converged, best = iterate_to_convergence(ret)

        valids.append(valid)
        convergeds.append(converged)
        iterations.append(best)
        states.append(ret.μs[best].flatten())

        i += 1

        #if i > 20: break

    # State vector output
    valids = pd.Series(valids, name="valid")
    pd.DataFrame(np.vstack([convergeds, iterations]).T, columns=["converged", "iterations"],
            index=valids).to_csv("../data/unified/retrievals/convergence" + name + ".csv")
    pd.DataFrame(np.vstack(states), columns=prior_means.columns, index=valids
            ).to_csv("../data/unified/retrievals/x" + name + ".csv")

