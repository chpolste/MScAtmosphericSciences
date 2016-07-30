import numpy as np
from scipy.integrate import cumtrapz


class USStandardBackground:
    """Static profile extension with the US Standard atmosphere.
    
    Humidity currently fixed at zero.
    """

    def __init__(self, lower, upper, *, p0=None, n=5000):
        """"""
        self.z = np.linspace(lower, upper, n)
        self.T = self._get_temperature()
        self.p = self._get_pressure(p0=p0)

    def evaluate(self, absorption, angle=0., cosmic=2.75):
        """
        
        absorption must be callable with arguments (θ=300/T, pd, e).
        """
        cosangle = np.cos(np.deg2rad(angle))
        α = absorption(θ=300/self.T, pd=self.p, e=0.)
        τexp = np.exp(-cumtrapz(α, self.z, initial=0)/cosangle)
        return cosmic * τexp[-1] + np.trapz(α * self.T * τexp, self.z)/cosangle

    def _get_temperature(self, z=None):
        if z is None: z = self.z
        assert z[-1] < 47000
        T = np.zeros_like(z)
        # Upper stratosphere
        T[z<47000] = 228.65 + 0.0028*(z[z<47000]-32000)
        # Lower stratosphere
        T[z<32000] = 216.65 + 0.001*(z[z<32000]-20000)
        # Tropopause
        T[z<20000] = 216.65
        # Troposphere
        T[z<11000] = 288.15 - 0.0065*z[z<11000]
        return T

    def _get_pressure(self, z=None, p0=None):
        T = self._get_temperature(z=z)
        if z is None: z = self.z
        # If no pressure for lower height is given, calculate from profile
        if p0 is None:
            zz = np.linspace(0, z[0], 5000)
            p0 = self._get_pressure(z=zz, p0=1013.25)[-1]
        # Hydrostatic equilibrium
        p = p0 * np.exp(-9.8076 * cumtrapz(1/(287*T), z, initial=0))
        return p
