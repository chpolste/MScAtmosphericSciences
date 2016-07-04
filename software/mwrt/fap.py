"""Fast absorption predictor with full forward differentiation.

Murphy, D. M., and T. Koop, 2005: Review of the vapour pressures of ice and
    supercooled water for atmospheric applications. Q. J. R. Meteorol. Soc.,
    131, 1539–1565, doi:10.1256/qj.04.94.
"""

import abc

import numpy as np
import scipy.sparse as sp

from mwrt.model import Value


class VectorValue(Value):
    """Convenience container for forward mode automatic differentiation."""

    def __mul__(self, other):
        """Product rule of derivation."""
        assert isinstance(other, VectorValue)
        return VectorValue( 
                fwd = self.fwd * other.fwd,
                dT = self.dT*other.fwd + self.fwd*other.dT,
                dlnq = self.dlnq*other.fwd + self.fwd*other.dlnq
                )

    def __truediv__(self, other):
        """Quotient rule of derivation."""
        assert isinstance(other, VectorValue)
        return VectorValue( 
                fwd = self.fwd / other.fwd,
                dT = (self.dT*other.fwd-self.fwd*other.dT) / other.fwd**2,
                dlnq = (self.dlnq*other.fwd-self.fwd*other.dlnq) / other.fwd**2
                )

    def __add__(self, other):
        assert isinstance(other, VectorValue)
        return VectorValue( 
                fwd = self.fwd + other.fwd,
                dT = self.dT + other.dT,
                dlnq = self.dlnq + other.dlnq
                )

    def __sub__(self, other):
        assert isinstance(other, VectorValue)
        return VectorValue( 
                fwd = self.fwd - other.fwd,
                dT = self.dT - other.dT,
                dlnq = self.dlnq - other.dlnq
                )


def qtot(lnq):
    """Total specific water content from lnq."""
    out = np.exp(lnq)
    return VectorValue(fwd=out, dT=0., dlnq=out)


def density(p, T):
    """Density of air according to ideal gas law (ρ = p/R/T).
    
    The specific gas constant is chosen as 288 J/kg/K. The additional 1 J/kg/K
    relative to dry air takes into account water vapor (which has 461.5 J/kg/K)
    without the need to calculate e explicitly. The errors of this
    approximation should be below 2 %.
    """
    out = p * 100 / 288 / T # p is given in hPa
    return VectorValue(fwd=out, dT=-out/T, dlnq=0.)


def esat(T):
    """Saturation water vapor pressure over water.
    
    Formula taken from Murphy and Koop (2005) who have taken it from the US
    Meteorological Handbook (1997). It is supposed to be valid for
    temperatures down to -50°C.
    """
    TT = T - 32.18
    out = 6.1121 * np.exp(17.502 * (T-273.15) / TT)
    return VectorValue(fwd=out, dT=4217.45694/TT/TT*out, dlnq=0.)


def qsat(p, T):
    """Saturation specific humidity."""
    esat_ = esat(T)
    factor = 0.622/p
    return VectorValue(
            fwd = factor * esat_.fwd,
            dT = factor * esat_.dT,
            dlnq = factor * esat_.dlnq
            )


def rh(qtot, qsat):
    """Fraction of total specific water and saturation specific humidity."""
    return qtot / qsat


def partition_lnq(p, T, lnq):
    """Separate specific water into vapor and liquid components."""
    qtot_ = qtot(lnq)
    qsat_ = qsat(p, T)
    rh_ = rh(qtot_, qsat_)
    mid = (0.95 <= rh_.fwd) & (rh_.fwd <= 1.05)
    high = 1.05 < rh_.fwd
    # Calculate partition function
    fliq = np.zeros_like(p)
    fliq_drh = np.zeros_like(p)
    # 0.95 <= s <= 1.05: transition into cloud starting at RH = 95%
    fliq[mid] = 0.5*(rh_.fwd[mid]-0.95-0.1/np.pi*np.cos(10*np.pi*rh_.fwd[mid]))
    fliq_drh[mid] = np.cos(5*np.pi*(rh_.fwd[mid]-1.05))**2
    # s > 1.05: RH is capped at 100% from here on only cloud water increases
    fliq[high] = rh_.fwd[high] - 1.
    fliq_drh[high] = 1.
    # Multiply with qsat to obtain specific amount
    qliq = VectorValue(
            fwd = qsat_.fwd * fliq,
            #             product rule         (  chain rule     )
            dT = qsat_.dT * fliq + qsat_.fwd * (rh_.dT * fliq_drh),
            dlnq = qsat_.dlnq * fliq + qsat_.fwd * (rh_.dlnq * fliq_drh),
            )
    # All water that's not liquid is vapor
    return qtot_ - qliq, qliq


class FastAbsorptionPredictor(metaclass=abc.ABCMeta):
    """"""

    def __call__(self, p, T, lnq):
        """"""
        qvap, qliq = partition_lnq(p, T, lnq)
        density_ = density(p, T),
        absorp_gas = self.gaseous_FAP(p, T, qvap)
        absorp_clw = self.cloud_FAP(p, T, lnq) * qliq / density_
        out = absorp_gas + absorp_clw
        return VectorValue(out.fwd, sp.diags(out.dT), spdiags(out.dlnq))

    @staticmethod
    @abc.abstractmethod
    def cloud_FAP(self, T):
        """"""

    @staticmethod
    @abc.abstractmethod
    def gaseous_FAP(self, p, T, lnq):
        """"""

