"""Optimal estimation tools and configuration.

Naming conventions:
x, μ - state vector
y - radiometer observation
"""

import numpy as np
from scipy.integrate import cumtrapz

from mwrt import MWRTM, LinearInterpolation
from faps_hatpro import *


# Retrieval grid
z_hatpro = 612.
zgrid = np.logspace(np.log10(z_hatpro), np.log10(15000.), 50).astype(int).astype(float)


class VirtualHATPRO:

    absorp = [
            FAP22240MHz, FAP23040MHz, FAP23840MHz, FAP25440MHz, FAP26240MHz,
            FAP27840MHz, FAP31400MHz, FAP51260MHz, FAP52280MHz, FAP53860MHz,
            FAP54940MHz, FAP56660MHz, FAP57300MHz, FAP58000MHz
            ]

    angles = [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8]

    def __init__(self, z_retrieval, z_model, model_error,
            scanning=(10, 11, 12, 13)):
        """

        Use only zenith for K band and three most transparent channels of
        V band but all angles for four most opaque channels of V band.
        """
        self.z = z_retrieval
        itp = LinearInterpolation(source=z_retrieval, target=z_model)
        state_dims = 0
        self.mod_ang = []
        for i, a in enumerate(self.absorp):
            angles = self.angles if i in scanning else [0.]
            self.mod_ang.append([MWRTM(itp, a), angles])
            state_dims += len(angles)
        self.model_error = model_error
        assert state_dims == len(self.model_error)

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
    
    def retrieve(self, y, μ0, p0, prior, γ0=5.0e3, max_iterations=15):
        """Levenberg Marquard minimization using form 5.36 from Rodgers (2000)"""
        μ, cov = μ0, prior.cov
        γ, dist = γ0, 1.0e20
        me = self.model_error
        for i in range(15):
            Fμ, jac = self.simulate(μ, p0)
            rhs = jac.T @ me.covi @ (y - Fμ - me.mean) + prior.covi @ (μ - prior.mean)
            cov = prior.covi + jac.T @ me.covi @ jac
            lhs = cov + γ * prior.covi
            diff = np.linalg.solve(lhs, rhs)        
            μ = μ + diff
            # Convergence check
            dist_old = dist
            dist = diff.T @ np.linalg.inv(cov) @ diff
            if dist < 2/γ: # how to do this best...? gamma influences dist
                return μ, cov
            # No convergence, adjust γ, try again
            if dist/dist_old > 0.75:
                γ = γ * 5
            elif dist/dist_old < 0.25:
                γ = γ * 0.5
        raise StopIteration()


class Gaussian:
    """Gaussian distribution with convenience methods/properties."""
    
    def __init__(self, mean, cov):
        self.mean = np.array(mean).reshape(-1,1)
        self.cov = np.array(cov)
        assert self.mean.shape[0] == self.cov.shape[0] == self.cov.shape[1]
    
    def sample(self, size):
        return np.random.multivariate_normal(mean=self.mean.flatten(), cov=self.cov, size=size)
    
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

