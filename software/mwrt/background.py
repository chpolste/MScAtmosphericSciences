import numpy as np
from scipy.integrate import cumtrapz


class USStandardBackground:
    """Static profile extension with the US Standard atmosphere.
    
    Humidity currently fixed at zero.
    """

    def __init__(self, lower, upper, p0, n=3000):
        """"""
        assert upper < 47000
        z = np.linspace(lower, upper, n)
        T = np.zeros_like(z)
        # Upper stratosphere
        T[z<47000] = 228.65 + 0.0028*(z[z<47000]-32000)
        # Lower stratosphere
        T[z<32000] = 216.65 + 0.001*(z[z<32000]-20000)
        # Tropopause
        T[z<20000] = 216.65
        # Troposphere
        T[z<11000] = 288.15 - 0.0065*z[z<11000]
        # Hydrostatic equilibrium
        p = p0 * np.exp(-9.8076 * cumtrapz(1/(287*T), z, initial=0))
        self.z, self.p, self.T = z, p, T

    def evaluate(self, absorption, angle=0., cosmic=2.75):
        """
        
        absorption must be callable with arguments (300/T, p, e).
        """
        cosangle = np.cos(np.deg2rad(angle))
        α = absorption(300/self.T, self.p, 0.)
        τexp = np.exp(-cumtrapz(α, self.z, initial=0)/cosangle)
        return cosmic * τexp[-1] + np.trapz(α * self.T * τexp, self.z)/cosangle

