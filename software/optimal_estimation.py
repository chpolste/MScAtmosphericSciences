"""Optimal estimation tools and configuration.

Naming conventions:
x - state vector
y - radiometer observation
"""

import numpy as np
from scipy.integrate import cumtrapz

from mwrt import MWRTM, LinearInterpolation
from faps_hatpro import *

# Retrieval grid
z_hatpro = 612.
zgrid = np.logspace(np.log10(z_hatpro), np.log10(15000.), 50).astype(int).astype(float)


class VirtualHATPRO:

    absorp = [
            FAP22240MHz, FAP23040MHz, FAP23840MHz, FAP25440MHz, FAP26240MHz,
            FAP27840MHz, FAP31400MHz, FAP51260MHz, FAP52280MHz, FAP53860MHz,
            FAP54940MHz, FAP56660MHz, FAP57300MHz, FAP58000MHz
            ]

    angles = [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8]

    def __init__(self, z, zgrid=zgrid):
        """

        Use only zenith for K band and three most transparent channels of
        V band but all angles for four most opaque channels of V band.
        """
        self.z = z
        itp = LinearInterpolation(source=z, target=zgrid)
        self.mod_ang = []
        for a in self.absorp[:10]:
            self.mod_ang.append([MWRTM(itp, a), self.angles[0]])
        for a in self.absorp[10:]:
            self.mod_ang.append([MWRTM(itp, a), self.angles])

    def separate(self, x, p0):
        """Take apart the state vector and calculate pressure.
        
        Approximate pressure by barometric height formula. The specific gas
        constant is set to 288 to account for water vapor. The problem is that
        a good estimation of the actual R requires qvap but partition_lnq
        requires p to determine qsat, so there is a circular dependency. The
        assumption of 288 is made in the density calculation of the FAP as well
        and the error is small enough that an iterative procedure for
        determining R is unnecessary.
        """
        n = len(x)
        T, lnq = x[:n], x[n:]
        p = p0 * np.exp(-9.8076 * cumtrapz(1/(288*T), self.z, initial=0))
        return p, T, lnq

    def simulate(self, x, p0, only_forward=False):
        """Calculate brightness temperatures and Jacobian."""
        p, T, lnq = self.separate(x, p0)
        fwd, jac = [], []
        for model, angles in self.mod_ang:
            if only_forward:
                result = model.forward(angles=angles, p=p, T=T, lnq=lnq)
            else:
                result = model(angles=angles, p=p, T=T, lnq=lnq)
                jac.append(np.hstack([result.dT, result.dlnq]))
            fwd.append(result.fwd)
        fwd = np.vstack(fwd)
        jac = np.vstack(jac)
        return fwd if only_fwd else (fwd, jac)




