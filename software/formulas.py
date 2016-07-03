"""Meteorological constants and formulas.

Variable    Unit        Description
-----------------------------------------------------------------
p           *hPa        Pressure
pd          *hPa        Dry air partial pressure
e           *hPa        Water vapor partial pressure
esat        *hPa        Water vapor saturation pressure
T           K           Temperature
Td          K           Dew point temperature
Tpot        K           Potential temperature
RH          %           Relative humidity
r           kg/kg       Water vapor mixing ratio
qvap        kg/kg       Specific humidity of water vapor
qliq        kg/kg       Specific humidity of liquid water
qcloud      kg/kg       Specific humidity of liquid and solid water
qsat        kg/kg       Saturation specific humidity
ρ           kg/m³       Density
Lvap        J/kg/K      Latent heat of evaporation

* marks non-standard uses of units.


References
----------
Hewison, T. J. (2006). Profiling Temperature and Humidity by Ground-based
    Microwave Radiometers. University of Reading.
Karstens, U., Simmer, C., & Ruprecht, E. (1994). Remote sensing of cloud liquid
    water. Meteorology and Atmospheric Physics, 54(1-4), 157–171.
"""

import inspect
from collections import OrderedDict

import numpy as np
from scipy.integrate import cumtrapz


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
c0 = 299792458. # m/s

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
RH.register(lambda e, esat: 100 * e/esat)
RH.register(lambda T, e: RH(e=e, esat=esat(T=T)))
RH.register(lambda T, Td: RH(e=esat(T=Td), esat=esat(T=T)))

# Dry air partial pressure
pd = ArgNameDispatch("pd")
pd.register(lambda p, e: p - e)

# Water vapor partial pressure
e = ArgNameDispatch("e")
e.register(lambda RH, esat: esat * RH/100)
e.register(lambda Td: esat(T=Td)) # e = saturation pressure at dew point

esat = ArgNameDispatch("esat")
@esat.register
def _(T): # Source: https://en.wikipedia.org/wiki/Goff%E2%80%93Gratch_equation
    return 10**(
            - 7.90298 * (373.15/T - 1)
            + 5.02808 * np.log10(373.15/T)
            - 1.3816e-7 * (10**(11.344*(1-T/373.15)) - 1)
            + 8.1328e-3 * (10**(-3.49149*(373.15/T-1)) - 1)
            + np.log10(1013.246)
            )

# Density
ρ = ArgNameDispatch("ρ")
ρ.register(lambda p, T, e: ((p-e)/Rdry + e/Rwat)*100/T) # Ideal gas law

# Water vapor mixing ratio
r = ArgNameDispatch("r")
r.register(lambda p, e: 0.622 * e/(p-e))

# Specific humidity
qvap = ArgNameDispatch("qvap")
qvap.register(lambda p, e: 0.622 * e/p)
qvap.register(lambda p, Td: 0.622 * e(Td=Td)/p)
qvap.register(lambda p, T, RH: RH/100 * qsat(p=p, T=T))

qliq = ArgNameDispatch("qliq")
@qliq.register
def _(T, qcloud): # Hewison (2006), formula 4.7
    return qcloud * np.maximum(0, np.minimum((T-233)/40, 1))
@qliq.register
def _(z, p, T, Td): # Adiabatic LWC model by Karstens et al. (1994)
    zindex = z.index if hasattr(z, "index") else None
    ztype = type(z)
    z = np.array(z)
    p = np.array(p)
    T = np.array(T)
    Td = np.array(Td)
    def cloud_lwc(z, p, ρ, T, e):
        out = np.zeros_like(z, dtype=float)
        Lvap_ = Lvap(T=T)
        LR_ = g/cp - LRmoist(T=T, Lvap=Lvap_, r=r(p=p, e=e))
        out = cumtrapz(ρ * cp/Lvap_ * LR_, z, initial=0)
        out =  out * (1.239 - 0.145*np.log(z-z[0]))
        out[0] = 0
        return out
    RH_ = RH(T=T, Td=Td)
    clouds = []
    current = None
    for i in range(len(RH_)):
        if RH_[i] >= 95 and T[i] >= 253.15 and current is None:
            current = i
        elif (RH_[i] < 95 or T[i] < 253.15) and current is not None:
            clouds.append(slice(current, i))
            current = None
    e_ = e(Td=Td)
    ρ_ = ρ(p=p, T=T, e=e_)
    out = np.zeros_like(z, dtype=float)
    for cloud in clouds:
        out[cloud] = cloud_lwc(z[cloud], p[cloud], ρ_[cloud], T[cloud], e_[cloud])
    out[T<253.15] = 0
    out = out / ρ_
    if zindex is not None:
        out = ztype(out, index=zindex) # not pretty but eh
    return out

qsat = ArgNameDispatch("qsat")
qsat.register(lambda p, T: qvap(p=p, e=esat(T=T)))

# Latent heat of evaporation
Lvap = ArgNameDispatch("Lvap")
@Lvap.register
def _(T): # Source: https://en.wikipedia.org/wiki/Latent_heat
    T = T - celsius_offset
    return 2500800. - 2360.*T + 16.*T**2 - 0.06*T**3

LRmoist = ArgNameDispatch("LRmoist")
@LRmoist.register
def _(T, Lvap, r):
    """Moist adidabatic lapse rate.
    source: http://glossary.ametsoc.org/wiki/Moist-adiabatic_lapse_rate
    """
    numer = 1 + (Lvap * r)/(Rdry * T)
    denom = cp + (Lvap*Lvap * r)/(Rwat * T*T)
    return g * numer/denom
LRmoist.register(lambda T, r: LRmoist(T=T, Lvap=Lvap(T=T), r=r))

