"""Microwave region radiative transfer model for ground based applications."""


class MWRTM:
    """"""

    def __init__(self, fap, *, data=None, zs=None, ps=None, Ts=None, lnqs=None):
        """Precalculate absorption coefficients and their jacobian."""

    def simulate(self, angle):
        """Calculate brightness temperature and jacobian at z=0."""
        return brightness_temperature, jacobian_T, jacobian_lnq


class Radiometer:
    """"""

    def __init__(self, νs, angles, zs, fap):
        """"""

    def simulate(self, state):
        """"""
        return brightness_temperatures, jacobian

