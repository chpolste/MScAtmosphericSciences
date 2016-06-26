"""Microwave region radiative transfer model for ground based applications."""


from collections import namedtuple

import numpy as np
from scipy.integrate import trapz, cumtrapz


# Cache type

Cached = namedtuple("Cached", ["fwd", "jT", "jlnq"])


# Radiative Transfer Model

class MWRTM:
    """"""

    def __init__(self, *, fap, data=None, z=None, p=None, T=None, lnq=None):
        """
        
        Precalculates absorption coefficients and their jacobian.
        
        TODO: assert that z is monotonically increasing.
        """
        # Gather profile data
        if data is None: data = []
        assert "z" in data or z is not None
        assert "p" in data or p is not None
        assert "T" in data or T is not None
        assert "lnq" in data or lnq is not None
        self.z = np.array(data["z"]) if "z" in data else np.array(z)
        self.p = np.array(data["p"]) if "p" in data else np.array(p)
        self.T = np.array(data["T"]) if "T" in data else np.array(T)
        self.lnq = np.array(data["lnq"]) if "lnq" in data else np.array(lnq)
        # Absorption prediction
        self.fap = fap
        # Setup cache
        self._cache_absorption()
        self._cache_optical_depth()

    def forward(self, angle):
        """Angle in degrees"""
        iangle = 1/np.cos(np.deg2rad(angle))
        # TODO: cosmic background
        return trapz(self.α.fwd * self.T * np.exp(iangle*self.τ.fwd), self.z)

    def jacobian_T(self, angle):
        """"""

    def jacobian_lnq(self, angle):
        """"""

    def _cache_absorption(self):
        """Calculate absorption coefficients with FAP and store in cache."""
        self.α = Cached(
                self.fap.forward(self.p, self.T, self.lnq),
                self.fap.jacobian_T(self.p, self.T, self.lnq),
                self.fap.jacobian_lnq(self.p, self.T, self.lnq)
                )

    def _cache_optical_depth(self):
        """"""
        self.τ = Cached(
                -cumtrapz(self.α.fwd, self.z, initial=0),
                -cumtrapz(self.α.jT, self.z, axis=1, initial=0),
                -cumtrapz(self.α.jlnq, self.z, axis=1, initial=0)
                )


class Radiometer:
    """"""

    def __init__(self, νs, faps):
        """"""

    def simulate(self, angles, state):
        """"""

