import inspect
from functools import partial

import numpy as np

from raso.constants import C


__all__ = ["get_var"]

__doc__ = """"""



class VarAccessor:
    """Variable accessor for Profiles.
    
    Knows how to calculate unknown quantities from known ones.
    """

    def __init__(self):
        self.registry = {}

    def register(self, output_var, f):
        if output_var not in self.registry:
            self.registry[output_var] = []
        input_vars = set(inspect.signature(f).parameters)
        self.registry[output_var].append((input_vars, f))

    def get_converters(self, available, target, exclude=None):
        """Find all converters that can produce the target from the availables

        Combinatorial blowup is somewhat mitigated by taking a use first match-
        strategy and removal of circular references. This part isn't very well
        tested though.
        """
        if exclude is None: exclude = set()
        if target in available: yield target, None
        if target not in self.registry: raise StopIteration()
        for input_vars, f in self.registry[target]:
            if len(input_vars & exclude) != 0:
                continue
            if input_vars <= available:
                yield input_vars, f
                continue
            # No immediate conversion possible, try nesting coversions.
            # To avoid circular references, the current target is added to
            # the excluded variables
            conv = partial(self.get_converters, available=available,
                    exclude=(exclude | {target}))
            for var in (input_vars - available):
                # Make sure there is a matching converter
                try: next(conv(target=var))
                except StopIteration:
                    break
            else:
                yield input_vars, f

    def __call__(self, profile, target):
        """Find a converter that calculates target from some/all of the
        quantities in available and apply it to the profile."""
        available = set(profile.variables)
        for input_vars, f in self.get_converters(available, target):
            if f is None: return profile.data[input_vars]
            return f(**{var: profile[var] for var in input_vars})
        raise KeyError("No matching accessor found.")


class VarAccessorDict:

    def __init__(self):
        self.accessor = VarAccessor()
        self.dct = {}

    def __getitem__(self, key):
        return self.dct[key]

    def __setitem__(self, key, val):
        if key.startswith("_"):
            self.dct[key] = val
        else:
            self.accessor.register(key, val)


class VarAccessorMeta(type):
    """Use class definition syntax for setting up a Converter object."""

    @classmethod
    def __prepare__(cls, name, bases):
        return VarAccessorDict()

    def __new__(cls, name, bases, dct):
        return dct.accessor


# Defined conversions:

class get_var(metaclass=VarAccessorMeta):
    """Metorological formulas."""

    def Tpot(T, p):
        """Potential temperature."""
        return T*(C.pref/p)**(C.Rdry/C.cp)

    def RH(e, es):
        """Relative humidity in %."""
        return 100. * e/es

    def es(T):
        """Saturation water vapor pressure.
        source: Federal Meteorological Handbook No. 3, formula D.3.1
                http://www.ofcm.gov/fmh3/pdf/00-entire-FMH3.pdf
        """
        T = T - C.celsius_offset
        return 611.21 * np.exp((17.502*T)/(240.97+T))

    def e(Td):
        """Water vapor pressure (= saturation pressure at dew point)."""
        Td = Td - C.celsius_offset
        return 611.21 * np.exp((17.502*Td)/(240.97+Td))

    def e(RH, es):
        """Water vapor pressure."""
        return es * RH/100.

    def density(T, p, e):
        """Density (according to ideal gas law)."""
        return ((p-e)/C.Rdry + e/C.Rwat)/T

    def r(p, e):
        """Water vapor mixing ratio.
        source: http://glossary.ametsoc.org/wiki/Mixing_ratio
        """
        return 0.622*e/(p-e)

    def Lvap(T):
        """Latent heat of evaporation.
        source: https://en.wikipedia.org/wiki/Latent_heat
        """
        T = T - C.celsius_offset
        return (2500800. - 2360.*T + 16.*T*T - 0.06*T**3)

    def LRmoist(T, Lvap, r):
        """Moist adidabatic lapse rate.
        source: http://glossary.ametsoc.org/wiki/Moist-adiabatic_lapse_rate 
        """
        numer = 1 + (Lvap * r)/(C.Rdry * T)
        denom = C.cp + (Lvap*Lvap * r)/(C.Rwat * T*T)
        return C.g * numer/denom

    #def LWC(RH, rho, z, Lvap):
    #    """Liquid water content as calculated by Karstens et al. (1994)."""
    #    lwcad = np.zeroslike(z)
    #    incloud, cloudbase = False, None
    #    for i in range(1, len(z)):
    #        if RH[i] >= 95 and T[i]:
    #            if not incloud: cloudbase = z[i]
    #            incloud = True
    #            lwcad = 
    #        else:
    #            incloud, cloudbase = False, None
    #    return lwcad*(1.239 - 0.145 * np.log(dh))

