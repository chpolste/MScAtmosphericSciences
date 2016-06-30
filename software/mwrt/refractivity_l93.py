"""Refractivity model by Liebe et al. (1993).

Provides models of refractivity by dry air, water vapor and liquid water.

Liebe, H. J., Hufford, G. A., & Cotton, M. G. (1993). Propagation modeling of
    moist air and suspended water/ice particles at frequencies below 1000 GHz.
    In Atmospheric Propagation Effects through Natural and Man-Made Obscurants
    for Visible through MM-Wave Radiation (pp. 3–1 – 3–11).
"""

from collections import namedtuple

import numpy as np


__all__ = ["refractivity_gaseous", "refractivity_lwc"]


def van_vleck_weisskopf(νeval, ν, γ, δ):
    """Complex Van Vleck-Weisskopf line shape function.

    νeval   GHz     frequency at which line shape is evaluated
    ν       GHz     center frequency of line
    γ       GHz     width parameter of line
    δ       -       overlap parameter of line

    Returned value is unitless.
    """
    term1 = (1. - 1j*δ)/(ν - νeval - 1j*γ)
    term2 = (1. + 1j*δ)/(ν + νeval + 1j*γ)
    return νeval * (term1 - term2)


# Dry-air module (2.2)

def refractivity_dry_nondispersive(ν, θ, pd, e):
    """Complex refractivity due to dry airnondispersive term.

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature
    pd      hPa     pressure of dry air
    e       hPa     pressure of water vapor

    Liebe et al. (1993).
    """
    return 0.2598e-6 * pd * θ


def refractivity_O2_lines(ν, θ, pd, e):
    """Complex refractivity due to O2 line terms.

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature
    pd      hPa     pressure of dry air
    e       hPa     pressure of water vapor

    Liebe et al. (1993).
    """
    N = 0.
    for line in O2_lines:
        # Line strength
        #S = line.a1 / line.ν * pd * θ**3 * np.exp(line.a2*(1-θ))
        S = line.a1 / line.ν * pd * θ**3 * np.exp(line.a2*(1-θ))
        # Width parameter (no mesosphere consideration)
        γ = line.a3 * (pd * θ**line.a4 + 1.10 * e * θ)
        γ = np.sqrt(γ*γ + (25*0.6e-4)**2)
        # Overlap parameter
        δ = (line.a5 + line.a6 * θ) * (pd+e) * θ**0.8
        # Refractivity
        N = N + S * van_vleck_weisskopf(ν, line.ν, γ, δ)
    return N


def refractivity_dry_continuum(ν, θ, pd, e):
    """Complex refractivity due to dry air continuum terms.

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature
    pd      hPa     pressure of dry air
    e       hPa     pressure of water vapor

    Liebe et al. (1993).
    """
    S0 = 6.14e-11 * pd * θ*θ
    γ0 = 0.56e-3 * (pd+e) * θ**0.8
    F0 = -ν/(ν + 1j*γ0)
    Sn = 1.40e-18 * pd*pd * θ**3.5
    Fn = ν/(1 + 1.9e-5*ν**1.5)
    return S0*F0 + 1j*Sn*Fn


# Water vapor module

def refractivity_H2O(ν, θ, pd, e):
    """Complex refractivity due to water vapor lines and continuum.

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature
    pd      hPa     pressure of dry air
    e       hPa     pressure of water vapor

    Liebe et al. (1993). The continuum term is included as a pseudo-line at
    1780 GHz.
    """
    N = 0.
    for line in H2O_lines:
        # Line strength
        S = line.b1 / line.ν * e * θ**3.5 * np.exp(line.b2*(1-θ))
        # Width parameter including approximation of Doppler broadening
        γ = line.b3 * (line.b4 * e * θ**line.b6 + pd * θ**line.b5)
        γDsq = (1.46 * line.ν)**2 / θ * 1.0e-12
        γ = 0.535 * γ + np.sqrt(0.217 * γ*γ + γDsq)
        # Refractivity
        N = N + S * van_vleck_weisskopf(ν, line.ν, γ, 0.0)
    return N


def refractivity_gaseous(ν, θ, pd, e):
    """Complex refractivity due to dry air and water vapor, lines and continuum.  

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature
    pd      hPa     pressure of dry air
    e       hPa     pressure of water vapor

    Liebe et al. (1993). Sum of dry nondispersive, O2 lines, dry continuum,
    H2O lines and H2O continuum terms.
    """
    return (  refractivity_dry_nondispersive(ν, θ, pd, e)
            + refractivity_O2_lines(ν, θ, pd, e)
            + refractivity_dry_continuum(ν, θ, pd, e)
            + refractivity_H2O(ν, θ, pd, e))


def refractivity_lwc(ν, θ):
    """Density specific complex refractivity due to liquid water.

    ν       GHz     frequency at which refractivity is evaluated
    θ       -       reciprocal temperature

    Returned is liquid density specific refractivity in m³/kg. Multiply by
    liquid water density to obtain unitless refractivity.

    Liebe et al. (1993).
    """
    ρL = 1000. # Density of liquid water kg/m³
    ε0 = 77.66 + 103.3 * (θ-1)
    ε1 = 0.0671 * ε0
    ε2 = 3.52
    γ1 = 20.20 - 146 * (θ-1) + 316 * (θ-1)**2
    γ2 = 39.8 * γ1
    εw = ε0 - ν * ((ε0 - ε1)/(ν + 1j*γ1) + (ε1 - ε2)/(ν + 1j*γ2))
    # Density specific refractivity
    return 1.5 / ρL * (εw - 1)/(εw + 2)


# Spectral line database

O2Line = namedtuple("O2Line", ["ν", "a1", "a2", "a3", "a4", "a5", "a6"])
O2_lines = [
        #             GHz    GHz/hPa          GHz/hPa           1/hPa      1/hPa
        O2Line( 50.474238, 9.400e-14, 9.694, 8.90e-04, 0.8,  2.40e-04,  7.90e-04),
        O2Line( 50.987749, 2.460e-13, 8.694, 9.10e-04, 0.8,  2.20e-04,  7.80e-04),
        O2Line( 51.503350, 6.080e-13, 7.744, 9.40e-04, 0.8,  1.97e-04,  7.74e-04),
        O2Line( 52.021410, 1.414e-12, 6.844, 9.70e-04, 0.8,  1.66e-04,  7.64e-04),
        O2Line( 52.542394, 3.102e-12, 6.004, 9.90e-04, 0.8,  1.36e-04,  7.51e-04),
        O2Line( 53.066907, 6.410e-12, 5.224, 1.02e-03, 0.8,  1.31e-04,  7.14e-04),
        O2Line( 53.595749, 1.247e-11, 4.484, 1.05e-03, 0.8,  2.30e-04,  5.84e-04),
        O2Line( 54.130000, 2.280e-11, 3.814, 1.07e-03, 0.8,  3.35e-04,  4.31e-04),
        O2Line( 54.671159, 3.918e-11, 3.194, 1.10e-03, 0.8,  3.74e-04,  3.05e-04),
        O2Line( 55.221367, 6.316e-11, 2.624, 1.13e-03, 0.8,  2.58e-04,  3.39e-04),
        O2Line( 55.783802, 9.535e-11, 2.119, 1.17e-03, 0.8, -1.66e-04,  7.05e-04),
        O2Line( 56.264775, 5.489e-11, 0.015, 1.73e-03, 0.8,  3.90e-04, -1.13e-04),
        O2Line( 56.363389, 1.344e-10, 1.660, 1.20e-03, 0.8, -2.97e-04,  7.53e-04),
        O2Line( 56.968206, 1.763e-10, 1.260, 1.24e-03, 0.8, -4.16e-04,  7.42e-04),
        O2Line( 57.612484, 2.141e-10, 0.915, 1.28e-03, 0.8, -6.13e-04,  6.97e-04),
        O2Line( 58.323877, 2.386e-10, 0.626, 1.33e-03, 0.8, -2.05e-04,  5.10e-05),
        O2Line( 58.446590, 1.457e-10, 0.084, 1.52e-03, 0.8,  7.48e-04, -1.46e-04),
        O2Line( 59.164207, 2.404e-10, 0.391, 1.39e-03, 0.8, -7.22e-04,  2.66e-04),
        O2Line( 59.590983, 2.112e-10, 0.212, 1.43e-03, 0.8,  7.65e-04, -9.00e-05),
        O2Line( 60.306061, 2.124e-10, 0.212, 1.45e-03, 0.8, -7.05e-04,  8.10e-05),
        O2Line( 60.434776, 2.461e-10, 0.391, 1.36e-03, 0.8,  6.97e-04, -3.24e-04),
        O2Line( 61.150560, 2.504e-10, 0.626, 1.31e-03, 0.8,  1.04e-04, -6.70e-05),
        O2Line( 61.800154, 2.298e-10, 0.915, 1.27e-03, 0.8,  5.70e-04, -7.61e-04),
        O2Line( 62.411215, 1.933e-10, 1.260, 1.23e-03, 0.8,  3.60e-04, -7.77e-04),
        O2Line( 62.486260, 1.517e-10, 0.083, 1.54e-03, 0.8, -4.98e-04,  9.70e-05),
        O2Line( 62.997977, 1.503e-10, 1.665, 1.20e-03, 0.8,  2.39e-04, -7.68e-04),
        O2Line( 63.568518, 1.087e-10, 2.115, 1.17e-03, 0.8,  1.08e-04, -7.06e-04),
        O2Line( 64.127767, 7.335e-11, 2.620, 1.13e-03, 0.8, -3.11e-04, -3.32e-04),
        O2Line( 64.678903, 4.635e-11, 3.195, 1.10e-03, 0.8, -4.21e-04, -2.98e-04),
        O2Line( 65.224071, 2.748e-11, 3.815, 1.07e-03, 0.8, -3.75e-04, -4.23e-04),
        O2Line( 65.764772, 1.530e-11, 4.485, 1.05e-03, 0.8, -2.67e-04, -5.75e-04),
        O2Line( 66.302091, 8.009e-12, 5.225, 1.02e-03, 0.8, -1.68e-04, -7.00e-04),
        O2Line( 66.836830, 3.946e-12, 6.005, 9.90e-04, 0.8, -1.69e-04, -7.35e-04),
        O2Line( 67.369598, 1.832e-12, 6.845, 9.70e-04, 0.8, -2.00e-04, -7.44e-04),
        O2Line( 67.900867, 8.010e-13, 7.745, 9.40e-04, 0.8, -2.28e-04, -7.53e-04),
        O2Line( 68.431005, 3.300e-13, 8.695, 9.20e-04, 0.8, -2.40e-04, -7.60e-04),
        O2Line( 68.960311, 1.280e-13, 9.695, 9.00e-04, 0.8, -2.50e-04, -7.65e-04),
        O2Line(118.750343, 9.450e-11, 0.009, 1.63e-03, 0.8, -3.60e-05,  9.00e-06),
        O2Line(368.498350, 6.790e-12, 0.049, 1.92e-03, 0.2,  0.00e+00,  0.00e+00),
        O2Line(424.763124, 6.380e-11, 0.044, 1.93e-03, 0.2,  0.00e+00,  0.00e+00),
        O2Line(487.249370, 2.350e-11, 0.049, 1.92e-03, 0.2,  0.00e+00,  0.00e+00),
        O2Line(715.393150, 9.960e-12, 0.145, 1.81e-03, 0.2,  0.00e+00,  0.00e+00),
        O2Line(773.839675, 6.710e-11, 0.130, 1.82e-03, 0.2,  0.00e+00,  0.00e+00),
        O2Line(834.145330, 1.800e-11, 0.147, 1.81e-03, 0.2,  0.00e+00,  0.00e+00)
        ]

H2OLine = namedtuple("H2OLine", ["ν", "b1", "b2", "b3", "b4", "b5", "b6"])
H2O_lines = [
        #               GHz    GHz/hPa            GHz/hPa
        H2OLine(  22.235080, 1.130e-08,  2.143, 2.811e-03,  4.80, 0.69, 1.00),
        H2OLine(  67.803960, 1.200e-10,  8.735, 2.858e-03,  4.93, 0.69, 0.82),
        H2OLine( 119.995940, 8.000e-11,  8.356, 2.948e-03,  4.78, 0.70, 0.79),
        H2OLine( 183.310091, 2.420e-07,  0.668, 3.050e-03,  5.30, 0.64, 0.85),
        H2OLine( 321.225644, 4.830e-09,  6.181, 2.303e-03,  4.69, 0.67, 0.54),
        H2OLine( 325.152919, 1.499e-07,  1.540, 2.783e-03,  4.85, 0.68, 0.74),
        H2OLine( 336.222601, 1.100e-10,  9.829, 2.693e-03,  4.74, 0.69, 0.61),
        H2OLine( 380.197372, 1.152e-06,  1.048, 2.873e-03,  5.38, 0.54, 0.89),
        H2OLine( 390.134508, 4.600e-10,  7.350, 2.152e-03,  4.81, 0.63, 0.55),
        H2OLine( 437.346667, 6.500e-09,  5.050, 1.845e-03,  4.23, 0.60, 0.48),
        H2OLine( 439.150812, 9.218e-08,  3.596, 2.100e-03,  4.29, 0.63, 0.52),
        H2OLine( 443.018295, 1.976e-08,  5.050, 1.860e-03,  4.23, 0.60, 0.50),
        H2OLine( 448.001075, 1.032e-06,  1.405, 2.632e-03,  4.84, 0.66, 0.67),
        H2OLine( 470.888947, 3.297e-08,  3.599, 2.152e-03,  4.57, 0.66, 0.65),
        H2OLine( 474.689127, 1.262e-07,  2.381, 2.355e-03,  4.65, 0.65, 0.64),
        H2OLine( 488.491133, 2.520e-08,  2.853, 2.602e-03,  5.04, 0.69, 0.72),
        H2OLine( 503.568532, 3.900e-09,  6.733, 1.612e-03,  3.98, 0.61, 0.43),
        H2OLine( 504.482692, 1.300e-09,  6.733, 1.612e-03,  4.01, 0.61, 0.45),
        H2OLine( 547.676440, 9.701e-07,  0.114, 2.600e-03,  4.50, 0.70, 1.00),
        H2OLine( 552.020960, 1.477e-06,  0.114, 2.600e-03,  4.50, 0.70, 1.00),
        H2OLine( 556.936002, 4.874e-05,  0.159, 3.210e-03,  4.11, 0.69, 1.00),
        H2OLine( 620.700807, 5.012e-07,  2.200, 2.438e-03,  4.68, 0.71, 0.68),
        H2OLine( 645.866155, 7.130e-09,  8.580, 1.800e-03,  4.00, 0.60, 0.50),
        H2OLine( 658.005280, 3.022e-08,  7.820, 3.210e-03,  4.14, 0.69, 1.00),
        H2OLine( 752.033227, 2.396e-05,  0.396, 3.060e-03,  4.09, 0.68, 0.84),
        H2OLine( 841.053973, 1.400e-09,  8.180, 1.590e-03,  5.76, 0.33, 0.45),
        H2OLine( 859.962313, 1.472e-08,  7.989, 3.060e-03,  4.09, 0.68, 0.84),
        H2OLine( 899.306675, 6.050e-09,  7.917, 2.985e-03,  4.53, 0.68, 0.90),
        H2OLine( 902.616173, 4.260e-09,  8.432, 2.865e-03,  5.10, 0.70, 0.95),
        H2OLine( 906.207325, 1.876e-08,  5.111, 2.408e-03,  4.70, 0.70, 0.53),
        H2OLine( 916.171582, 8.340e-07,  1.442, 2.670e-03,  4.78, 0.70, 0.78),
        H2OLine( 923.118427, 8.690e-09, 10.220, 2.900e-03,  5.00, 0.70, 0.80),
        H2OLine( 970.315022, 8.972e-07,  1.920, 2.550e-03,  4.94, 0.64, 0.67),
        H2OLine( 987.926764, 1.321e-05,  0.258, 2.985e-03,  4.55, 0.68, 0.90),
        # Pseudo-line that represents continuum:
        H2OLine(1780.000000, 2.230e-03,  0.952, 1.762e-02, 30.50, 2.00, 5.00)
        ]
