"""Microwave region radiative transfer model for ground based applications."""

from numbers import Number

import numpy as np
import scipy.integrate as spi

from mwrt.autodiff import Vector, DiagVector, exp, trapz, cumtrapz


__all__ = ["MWRTM", "Radiometer"]


class MWRTM:
    """Microwave radiative transfer model for ground based applications.
    
    Default mode of model (using __call__) uses full forward differentiation,
    providing Jacobians for temperature and humidity (input as the natural
    logarithm of specific water content, i.e vapor + liquid). The .forward
    method only calculates the brightness temperature without propagating
    derivatives (this is about 10 times faster than a run with
    differentiation).

    Model assumptions include:
    - Monochromatic light
    - Pencil beam
    - Horizontal homogeneity of atmosphere
    - Flatland world, i.e. no consideration of Earth's curvature
    - No refraction, i.e. bending of ray path is missing
    - ...
    """

    def __init__(self, interpolator, absorption):
        """Initialize a radiative transfer model.
        
        The interpolator is used to transform quantities from the input to the
        model grid.
        """
        self.interpolator = interpolator
        self.absorption = absorption

    def _get_vars(self, data, p, T, lnq):
        """Extract and check data input."""
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
        return p, T, lnq

    def __call__(self, *, data=None, p=None, T=None, lnq=None, angles=None):
        """Perform a model simulation.
        
        If angles is None, a Cached model will be returned which then can be
        called with angles to obtain the corresponding brightness temperatures.
        Angles are given in degrees deviation from zenith.
        """
        p, T, lnq = self._get_vars(data, p, T, lnq)
        cached_model = CachedMWRTM(self, p, T, lnq)
        if angles is None:
            return cached_model
        return cached_model(angles)

    def forward(self, angle, *, data=None, p=None, T=None, lnq=None):
        """Perform a model simulation without calculating the Jacobians."""
        p, T, lnq = self._get_vars(data, p, T, lnq)
        zz = self.interpolator.target
        pp = self.interpolator(p)
        TT = self.interpolator(T)
        qq = self.interpolator(lnq)
        α = self.absorption(pp, TT, qq)
        cosangle = np.cos(np.deg2rad(angle))
        τexp = np.exp(-spi.cumtrapz(α, zz, initial=0)/cosangle)
        cosmic = 2.736 * τexp[-1]
        return cosmic + np.trapz(α * TT * τexp, zz)/cosangle


class CachedMWRTM:
    """MWRTM with cached absorption and optical depth for faster evaluation."""

    def __init__(self, parent, p, T, lnq):
        zz = parent.interpolator.target
        pp = parent.interpolator(p)
        TT = parent.interpolator(T)
        qq = parent.interpolator(lnq)
        # Instead of propagating low-res dT and dlnq through entire FAP
        # calculation, the FAP jacobian is evaluated in high-res space (where
        # it is diagonal and much easier to handle).
        αα = parent.absorption(
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
                dT = αα.dT[:,None] * parent.interpolator.matrix,
                dlnq = αα.dlnq[:,None] * parent.interpolator.matrix
                )
        # Optical depth
        self.τ = -cumtrapz(self.α, zz, initial=0)
        # Keep vertical grid and temperature for evaluation
        self.z = zz
        self.T = Vector(
                fwd=TT,
                dT=parent.interpolator.matrix,
                dlnq=np.zeros_like(parent.interpolator.matrix, dtype=float)
                )

    def evaluate(self, angle):
        """Perform a model simulation.
        
        The angle is given in degrees deviation from zenith.
        """
        cosangle = np.cos(np.deg2rad(angle))
        τexp = exp(self.τ/cosangle)
        cosmic = 2.736 * τexp[-1]
        return cosmic + trapz(self.α * self.T * τexp, self.z)/cosangle

    def __call__(self, angles):
        """Perform a model simulation.
        
        Angles are given in degrees deviation from zenith.
        """
        if isinstance(angles, Number):
            return self.evaluate(angles)
        raise NotImplementedError()


class Radiometer:
    """"""

    def __init__(self, νs, faps):
        """"""

    def simulate(self, angles, state):
        """"""

