"""Turner, Keifel, Caddedu liquid water absorption model.

Turner, D. D., Kneifel, S., & Cadeddu, M. P. (2016). An Improved Liquid Water
    Absorption Model at Microwave Frequencies for Supercooled Liquid Water
    Clouds.  Journal of Atmospheric and Oceanic Technology, 33(1), 33–44.
    http://doi.org/10.1175/JTECH-D-15-0074.1
"""

import numpy as np


def refractivity_lwc(ν, θ):
    """Complex refractivity due to liquid water.

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature

    Returned is liquid density specific refractivity in m³/kg. Multiply by
    liquid water density to obtain unitless refractivity.

    Turner et al. (2016). Designed for frequency range from 0.5 to 500 GHz and
    temperatures from -40 °C to +50 °C.
    """
    ρL = 1000. # Density of liquid water kg/m³
    ν2pi = 2. * np.pi * ν
    ν2pisq = ν2pi * ν2pi
    # Formulas are valid for T in °C
    T = 300/θ - 273.15
    # Static dielectric permittivity
    ϵs = 8.7914e1 - 4.0440e-1 * T + 9.5873e-4 * T*T - 1.3280e-6 * T*T*T
    # Relaxation term components
    Δ1 = 8.111e1 * np.exp(-4.434e-3 * T)
    Δ2 = 2.025e0 * np.exp(-1.073e-2 * T)
    τ1 = 1.302e-13 * np.exp(6.627e2/(T + 1.342e2))
    τ2 = 1.012e-14 * np.exp(6.089e2/(T + 1.342e2))
    denom1 = 1 + ν2pisq*τ1*τ1
    denom2 = 1 + ν2pisq*τ2*τ2
    # Relaxation terms
    A1 = τ1*τ1*Δ1/denom1
    A2 = τ2*τ2*Δ2/denom2
    B1 = τ1*Δ1/denom1
    B2 = τ2*Δ2/denom2
    # Dielectric permittivity
    ϵreal = ϵs - ν2pisq * (A1 + A2)
    ϵimag = ν2pi * (B1 + B2)
    ϵ = ϵreal + 1j * ϵimag
    # Density specific refractivity
    return 1.5 / ρL * (ϵ - 1)/(ϵ + 2)
