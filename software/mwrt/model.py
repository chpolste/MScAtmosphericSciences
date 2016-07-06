"""Microwave region radiative transfer model for ground based applications."""

import numpy as np

from mwrt.autodiff import Vector, DiagVector, exp, trapz, cumtrapz


__all__ = ["MWRTM", "Radiometer"]


class MWRTM:
    """"""

    def __init__(self, interpolator, fap):
        """"""
        self.interpolator = interpolator
        self.fap = fap

    def __call__(self, *, data=None, p=None, T=None, lnq=None, angles=None):
        """"""
        if data is None: data = []
        assert "p" in data or p is not None
        assert "T" in data or T is not None
        assert "lnq" in data or lnq is not None
        p = np.array(data["p"]) if "p" in data else np.array(p)
        T = np.array(data["T"]) if "T" in data else np.array(T)
        lnq = np.array(data["lnq"]) if "lnq" in data else np.array(lnq)
        assert len(p) == len(self.interpolator.source)
        assert len(T) == len(self.interpolator.source)
        assert len(lnq) == len(self.interpolator.source)
        cached_model = CachedMWRTM(self, p, T, lnq)
        if angles is None:
            return cached_model
        return cached_model(angles)


class CachedMWRTM:
    """"""

    def __init__(self, parent, p, T, lnq):
        """"""
        zz = parent.interpolator.target
        pp = parent.interpolator @ p
        TT = parent.interpolator @ T
        qq = parent.interpolator @ lnq
        # Instead of propagating low-res dT and dlnq through entire FAP
        # calculation, the FAP jacobian is evaluated in high-res space (where
        # it is diagonal and much easier to handle).
        αα = parent.fap(
                DiagVector.init_p(pp),
                DiagVector.init_T(TT),
                DiagVector.init_lnq(qq)
                )
        # Obtain the Jacobian wrt the low-res data with the chain rule (the
        # interpolator is linear and therefore the interpolation matrix is also
        # its Jacobian). αα.dT[:,None] * itp performs the matmul with the
        # diagonal matrix αα.dT faster than using scipy.sparse.diags.
        self.α = Vector(
                fwd = αα.fwd,
                dT = alpha_fap.dT[:,None] * itp,
                dlnq = alpha_fap.dlnq[:,None] * itp
                )
        # Optical depth
        self.τ = -cumtrapz(self.α, zz, initial=0)
        # Keep vertical grid and temperature for evaluation
        self.z = zz
        self.T = TT

    def evaluate(self, angle):
        """Angle from zenith in degrees."""

    def __call__(self, angles):
        """Angle in degrees"""
        cosangle = np.cos(np.deg2rad(angle))
        τexp = np.exp(self.τ.fwd/cosangle) # cache this...?
        cosmic = 2.736 * τexp[-1] # TODO
        return cosmic + trapz(self.α.fwd * self.T * τexp, self.z)/cosangle


class Radiometer:
    """"""

    def __init__(self, νs, faps):
        """"""

    def simulate(self, angles, state):
        """"""

