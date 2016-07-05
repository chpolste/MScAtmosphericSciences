"""Fast absorption predictor with full forward differentiation.

Murphy, D. M., and T. Koop, 2005: Review of the vapour pressures of ice and
    supercooled water for atmospheric applications. Q. J. R. Meteorol. Soc.,
    131, 1539–1565, doi:10.1256/qj.04.94.
"""

import abc

import numpy as np
import scipy.sparse as sp

from mwrt.autodiff import VectorValue, exp


def density(p, T):
    """Density of air according to ideal gas law (ρ = p/R/T).
    
    The specific gas constant is chosen as 288 J/kg/K. The additional 1 J/kg/K
    relative to dry air takes into account water vapor (which has 461.5 J/kg/K)
    without the need to calculate e explicitly. The errors of this
    approximation should be below 2 %.
    """
    return (100./288.) * (p/T) # p is given in hPa

def esat(T):
    """Saturation water vapor pressure over water.
    
    Formula taken from Murphy and Koop (2005) who have taken it from the US
    Meteorological Handbook (1997). It is supposed to be valid for
    temperatures down to -50°C.
    """
    return 6.1121 * exp(17.502 * (T-273.15)/(T - 32.18))

def qsat(p, T):
    """Saturation specific humidity."""
    return 0.622 * esat(T) / p

def rh(qtot, qsat):
    """Fraction of total specific water and saturation specific humidity."""
    return qtot / qsat

def partition_lnq(p, T, lnq):
    """Separate specific water into vapor and liquid components.
    
    This is a piecewise function, the differentiation is carried out manually.
    """
    qtot = exp(lnq)
    qsat_ = qsat(p, T)
    rh_ = rh(qtot, qsat_)
    mid = (0.95 <= rh_.fwd) & (rh_.fwd <= 1.05)
    high = 1.05 < rh_.fwd
    # Calculate partition function
    fliq = np.zeros_like(p.fwd)
    fliq_drh = np.zeros_like(p.fwd)
    # 0.95 <= s <= 1.05: transition into cloud starting at RH = 95%
    fliq[mid] = 0.5*(rh_.fwd[mid]-0.95-0.1/np.pi*np.cos(10*np.pi*rh_.fwd[mid]))
    fliq_drh[mid] = np.cos(5*np.pi*(rh_.fwd[mid]-1.05))**2
    # s > 1.05: RH is capped at 100% from here on only cloud water increases
    fliq[high] = rh_.fwd[high] - 1.
    fliq_drh[high] = 1.
    # Multiply with qsat to obtain specific amount
    qliq = VectorValue(
            fwd = qsat_.fwd * fliq,
            #    (        product rule         (    chain rule   ))
            #    dqsat/dT * fliq + qsat *      dfliq/drh * drh/dT
            dT = qsat_.dT * fliq + qsat_.fwd * (fliq_drh * rh_.dT),
            dlnq = qsat_.dlnq * fliq + qsat_.fwd * (fliq_drh * rh_.dlnq)
            )
    # All water that's not liquid is vapor
    return qtot - qliq, qliq


class FastAbsorptionPredictor(metaclass=abc.ABCMeta):
    """"""

    def __call__(self, p, T, lnq):
        """"""
        qvap, qliq = partition_lnq(p, T, lnq)
        density_ = density(p, T)
        absorp_gas = self.gas_absorption(p, T, qvap)
        absorp_clw = self.cloud_absorption(T) * qliq * density_
        out = absorp_gas + absorp_clw
        return out
        return VectorValue(out.fwd, out.dT, out.dlnq)

    @staticmethod
    @abc.abstractmethod
    def cloud_absorption(self, T):
        """"""

    @staticmethod
    @abc.abstractmethod
    def gas_absorption(self, p, T, lnq):
        """"""

