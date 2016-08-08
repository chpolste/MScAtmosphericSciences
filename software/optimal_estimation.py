"""Optimal estimation tools and configuration.

Naming conventions:
x, μ - state vector
y - radiometer observation
"""

import numpy as np
from scipy.integrate import cumtrapz

from mwrt import MWRTM, LinearInterpolation
from faps_hatpro import faps, bgs


# Retrieval grid
z_hatpro = 612.
z_top = 12612.
# Retrieval grid
rgrid = np.round(np.logspace(np.log10(z_hatpro), np.log10(z_top), 50)).astype(float)
# Internal model grid
mgrid = np.logspace(np.log10(z_hatpro), np.log10(z_top), 2500)
# Parameter sequence
paramseq = (10000, 5000, 2500, 1000, 500, 250, 100, 50)


class Gaussian:
    """Gaussian distribution with convenience methods/properties."""

    def __init__(self, mean, cov):
        self.mean = np.array(mean).reshape(-1,1)
        self.cov = np.array(cov)
        assert self.mean.shape[0] == self.cov.shape[0] == self.cov.shape[1]

    def sample(self, size):
        return np.random.multivariate_normal(mean=self.mean.flatten(),
                cov=self.cov, size=size)

    @property
    def covi(self):
        """Memoized inverse of covariance."""
        if not hasattr(self, "_covi"):
            self._covi = np.linalg.inv(self.cov)
        return self._covi

    @classmethod
    def read_csv(cls, mean, cov):
        from db_tools import read_csv_covariance, read_csv_mean
        cov = read_csv_covariance(cov)
        if mean is None:
            mean = np.zeros(cov.shape[0])
        else:
            mean = read_csv_mean(mean)
        return cls(mean, cov)

    def __len__(self):
        return self.mean.shape[0]


class OptimalEstimationRetrieval:

    def __init__(self, *, model, params, y, p0, μ0, prior, obs_error):
        """Set up an optimal estimation retrieval.

        z: retrival grid
        model: forward model (accepts state vector and surface pressure,
            returns simulated observation and Jacobian)
        params: a sequence of parameters to control the Levenberg-Marquard
            minimization. Last value is repeated if sequence is too short.
        y: observation vector
        p0: surface pressure in hPa
        μ0: first guess of state vector
        prior: prior distribution of atmospheric state
        obs_error: observation/model error distribution
        """
        self.model = model
        self.params = list(params)
        self.y = y
        self.p0 = p0
        self.μs = [μ0]
        self.Fμs = [0]
        self.covs = [prior.cov]
        self.prior = prior
        self.obserr = obs_error
        self.counter = 0
        self.obs_measures = []
        self.state_measures = []
        self.costs = []

    def iterate(self, only=None, calculate_measures="all"):
        """Levenberg-Marquard step with 5.36 from Rodgers (2000).
        
        The 'only' parameter is just for test purposes. Use the specialized
        Virtual HATPROs instead.
        """
        μ = self.μs[-1]
        if len(self.params) > self.counter:
            γ = self.params[self.counter]
        else:
            γ = self.params[-1]
        Fμ, jac = self.model(μ, self.p0)
        rhs = (jac.T @ self.obserr.covi @ (self.y - Fμ - self.obserr.mean)
                + self.prior.covi @ (μ - self.prior.mean))
        covi = self.prior.covi + jac.T @ self.obserr.covi @ jac
        lhs = covi + γ * self.prior.covi
        diff = np.linalg.solve(lhs, rhs)
        # Save new values
        if only is not None:
            diff_ = diff
            diff = np.zeros_like(diff)
            diff[only] = diff_[only]
        self.μs.append(μ + diff)
        self.Fμs.append(Fμ)
        self.covs.append(np.linalg.inv(covi))
        self.counter += 1
        # Calculate requested convergence measures
        alldist = (calculate_measures == "all")
        if alldist or calculate_measures == "state":
            self.state_measures.append(float(diff.T @ covi @ diff))
        else:
            self.state_measures.append(None)
        if alldist or calculate_measures == "obs":
            m = (self.obserr.cov
                    @ np.linalg.inv(jac @ self.prior.cov @ jac.T + self.obserr.cov)
                    @ self.obserr.cov)
            d = self.Fμs[-2] - self.Fμs[-1]
            self.obs_measures.append(float(d.T @ m @ d))
        else:
            self.obs_measures.append(None)
        if alldist or calculate_measures == "cost":
            v1 = self.y - Fμ
            v2 = self.prior.mean - μ
            cost = v1.T @ self.obserr.covi @ v1 + v2.T @ self.prior.covi @ v2
            self.costs.append(float(cost))
        else:
            self.costs.append(None)


class VirtualHATPRO:

    absorptions = faps

    backgrounds = bgs

    angles = [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8]

    def __init__(self, z_retrieval, z_model, error, params,
            scanning=(10, 11, 12, 13)):
        """

        Use only zenith for K band and three most transparent channels of
        V band but all angles for four most opaque channels of V band.
        """
        self.z = z_retrieval
        itp = LinearInterpolation(source=z_retrieval, target=z_model)
        state_dims = 0
        self.mod_ang = []
        for i, (a, bg) in enumerate(zip(self.absorptions, self.backgrounds)):
            angles = self.angles if i in scanning else [0.]
            self.mod_ang.append([MWRTM(itp, a, background=bg), angles])
            state_dims += len(angles)
        self.error = error
        assert state_dims == len(self.error)
        self.params = params

    def separate(self, x, p0):
        """Take apart the state vector and calculate pressure.

        Approximate pressure by barometric height formula. The specific gas
        constant is set to 288 to account for water vapor. The problem is that
        a good estimation of the actual R requires qvap but partition_lnq
        requires p to determine qsat, so there is a circular dependency. The
        assumption of 288 is made in the density calculation of the FAP as well
        and the error is small enough that an iterative procedure for
        determining R is unnecessary.
        """
        n = x.shape[0]//2
        T, lnq = x[:n,:].flatten(), x[n:,:].flatten()
        p = p0 * np.exp(-9.8076 * cumtrapz(1/(288*T), self.z, initial=0))
        return p, T, lnq

    def simulate(self, x, p0, only_forward=False):
        """Calculate brightness temperatures and Jacobian."""
        p, T, lnq = self.separate(x, p0)
        fwd, jac = [], []
        for model, angles in self.mod_ang:
            if only_forward:
                result = model.forward(angles=angles, p=p, T=T, lnq=lnq)
            else:
                result = model(angles=angles, p=p, T=T, lnq=lnq)
                jac.append(np.hstack([result.dT, result.dlnq]))
            fwd.append(result.fwd)
        fwd = np.vstack(fwd)
        jac = np.vstack(jac)
        return fwd if only_forward else (fwd, jac)

    def retrieve(self, y, p0, μ0, prior, max_iterations=15, only=None):
        optest = OptimalEstimationRetrieval(
                model=self.simulate, params=self.params,
                y=y, p0=p0, μ0=μ0,
                prior=prior, obs_error=self.error
                )
        for i in range(max_iterations):
            optest.iterate(only=only)
        return optest


class VirtualHATPRO_Kband(VirtualHATPRO):
    
    absorptions = faps[:7]
    backgrounds = bgs[:7]
    angles = [0.]

    def __init__(self, z_retrieval, z_model, error, params, scanning=()):
        super().__init__(z_retrieval, z_model, error, params, scanning) 


class VirtualHATPRO_Vband(VirtualHATPRO):
    
    absorptions = faps[7:]
    backgrounds = bgs[7:]

    def __init__(self, z_retrieval, z_model, error, params, scanning=None):
        super().__init__(z_retrieval, z_model, error, params,
                scanning=(3, 4, 5, 6))

