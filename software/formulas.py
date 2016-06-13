"""Meteorological constants and formulas.

Variable    Unit        Description
-----------------------------------------------------------------
p           *hPa        Pressure
pd          *hPa        Dry air partial pressure
e           *hPa        Water vapor partial pressure
es          *hPa        Water vapor saturation pressure
T           K           Temperature
Td          K           Dew point temperature
Tpot        K           Potential temperature
RH          %           Relative humidity
r           kg/kg       Water vapor mixing ratio
density     kg/m³       Density
Lvap        J/kg/K      Latent heat of evaporation

* marks non-standard uses of units.
"""

import inspect
from collections import OrderedDict

import numpy as np


class ArgNameDispatch:
    """Generic function that dispatches on argument names.
    
    Current strategy for choosing a method is to pick the first match found.
    Use registry method as a function decorator to add methods.
    """

    def __init__(self, name):
        self.name = name
        self.registry = OrderedDict()
    
    def register(self, f):
        args_in = frozenset(inspect.signature(f).parameters)
        self.registry[args_in] = f
        return self

    def __call__(self, *args, **kwargs):
        # Auto-detect names of objects with a name attribute such as pd.Series
        for i, arg in enumerate(args):
            if hasattr(arg, "name"):
                if arg.name in kwargs.keys():
                    err = "Two values for {} are specified.".format(arg.name)
                    raise ValueError(err)
                kwargs[arg.name] = arg
            else:
                err = "Cannot process arg {}.".format(i)
                raise ValueError(err)
        # Find appropriate method to perform conversion
        available_args = set(kwargs.keys())
        for args_in, f in self.registry.items():
            if args_in <= available_args:
                out = f(**{arg: kwargs[arg] for arg in args_in})
                if hasattr(out, "name"): out.name = self.name
                return out
        # No method available
        err = "No method matching inputs {}".format(", ".join(kwargs.keys()))
        raise LookupError(err)

    def __repr__(self):
        if len(self.registry) == 0:
            return "ArgNameDispatch: {}. No methods.".format(self.name)
        out = ["ArgNameDispatch: {}. Methods:".format(self.name)]
        for i, args_in in enumerate(self.registry.keys(), start=1):
            arglist = ", ".join(args_in)
            out.append("{:>4}: {}({})".format(i, self.name, arglist))
        return "\n".join(out)


# Constants

g = 9.8076 # m/s²

# Dry air thermodynamics
Rdry = 287. # J/kg/K
cp = 1004. # J/kg/K
cv = 717. # J/kg/K
LRdry = 0.00977 # K/m (dry adiabatic lapse rate = g/cp)

# Moist air thermodynamics
Rwat = 461.5 # J/kg/K

# Latent heats
Lvap = 2.501e6 # J/kg
Lfus = 3.337e5 # J/kg
Lsub = 2.834e6 # J/kg

# Reference pressure for potential temperature
pref = 1000.00 # hPa

celsius_offset = 273.15 # K


# Formulas

# Potential temperature
Tpot = ArgNameDispatch("Tpot")
Tpot.register(lambda T, p: T*(pref/p)**(Rdry/cp))

# Relative humidity
RH = ArgNameDispatch("RH")
RH.register(lambda e, es: 100 * e/es)
RH.register(lambda T, Td: RH(e(Td=Td), es=es(T=T)))

# Dry air partial pressure
pd = ArgNameDispatch("pd")
pd.register(lambda p, e: p - e)

# Water vapor partial pressure
e = ArgNameDispatch("e")
e.register(lambda RH, es: es * RH/100)
e.register(lambda Td: es(T=Td)) # e = saturation pressure at dew point

es = ArgNameDispatch("es")
@es.register
def _(T): # Source: http://www.ofcm.gov/fmh3/pdf/00-entire-FMH3.pdf (D.3.1)
    T = T - celsius_offset
    return 611.21 * np.exp((17.502*T)/(240.97+T))

# Density
density = ArgNameDispatch("density")
density.register(lambda T, p, e: ((p-e)/Rdry + e/Rwat)/T) # Ideal gas law

# Water vapor mixing ratio
r = ArgNameDispatch("r")
r.register(lambda p, e: 0.622 * e/(p-e))

# Latent heat of evaporation
Lvap = ArgNameDispatch("Lvap")
@Lvap.register
def _(T): # Source: https://en.wikipedia.org/wiki/Latent_heat
    T = T - celsius_offset
    return 2500800. - 2360.*T + 16.*T**2 - 0.06*T**3
