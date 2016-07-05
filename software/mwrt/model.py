"""Microwave region radiative transfer model for ground based applications."""

import numpy as np
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d


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
        z = np.array(data["z"]) if "z" in data else np.array(z)
        p = np.array(data["p"]) if "p" in data else np.array(p)
        T = np.array(data["T"]) if "T" in data else np.array(T)
        lnq = np.array(data["lnq"]) if "lnq" in data else np.array(lnq)
        # Interpolate
        self.z = np.linspace(z[0], z[-1], 100000)
        self.p = interp1d(z, p)(self.z)
        self.T = interp1d(z, T)(self.z)
        self.lnq = interp1d(z, lnq)(self.z)
        # Absorption prediction
        self.fap = fap
        # Setup cache
        self._cache_absorption()
        self._cache_optical_depth()

    def forward(self, angle):
        """Angle in degrees"""
        cosangle = np.cos(np.deg2rad(angle))
        τexp = np.exp(self.τ.fwd/cosangle) # cache this...?
        cosmic = 2.736 * τexp[-1]
        return cosmic + trapz(self.α.fwd * self.T * τexp, self.z)/cosangle

    def jacobian_T(self, angle):
        """"""

    def jacobian_lnq(self, angle):
        """"""

    def _cache_absorption(self):
        """Calculate absorption coefficients with FAP and store in cache."""
        self.α = Value(
                self.fap.forward(self.p, self.T, self.lnq),
                None, #self.fap.jacobian_T(self.p, self.T, self.lnq),
                None #self.fap.jacobian_lnq(self.p, self.T, self.lnq)
                )

    def _cache_optical_depth(self):
        """"""
        self.τ = Value(
                -cumtrapz(self.α.fwd, self.z, initial=0),
                None, #-cumtrapz(self.α.jT, self.z, axis=1, initial=0),
                None, #-cumtrapz(self.α.jlnq, self.z, axis=1, initial=0)
                )


class Radiometer:
    """"""

    def __init__(self, νs, faps):
        """"""

    def simulate(self, angles, state):
        """"""

