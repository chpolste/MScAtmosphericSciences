__all__ = ["C"]

__doc__ = """Centralized definitions of physical constants."""


class C:

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
    pref = 100000 # Pa

    celsius_offset = 273.15 # K
