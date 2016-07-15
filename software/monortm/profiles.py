"""Tools to turn atmospheric profiles into their record representation.

MonoRTM takes gas amounts either as column density in molecules/cm² or
as molecular/volume mixing ratios in molecules/molecules. Internally the
two are separated by checking if the given value is smaller or larger than
one (monortm.f90, lines 421-422). Mixing ratios of all constituents are
relative to dry air.

Conversion between column density and mixing ratio is given by

    column density = mixing ratio · dz · p / k / T

The broadening gases in element 8 of record 2.1.2 must always be given as
a column density. I cannot find anywhere in the documentation what these
broadening gases are but it seems that they are the nobel gases since the
example profiles have mixing ratios of about 0.009 that are fairly constant
with height.
"""

from monortm.records import (Record21, Record211_IFORM0, Record211_IFORM1,
        Record212_first, Record212_other)

# Molecular/Volume mixing ratios
# Source: https://en.wikipedia.org/wiki/Atmosphere_of_Earth#Composition
mixing_ratio_N2 =  0.78084
mixing_ratio_O2 =  0.20946
mixing_ratio_Ar =  0.00934
mixing_ratio_CO2 = 0.00036 # Remaining parts

boltzmann = 1.3806485e-23
avogadro = 6.02214e23
Rdry = 287.
Rwat = 461.5


def layer(zs, ps, Ts, qvap, qliq, IFORM=1):
    """Create the records for an atmospheric layer.
    
    Contains only a minimal set of species. Make sure to set NMOL to 22.
    """
    assert IFORM == 0 or IFORM == 1
    assert len(zs) == 2
    assert len(ps) == 2
    assert len(Ts) == 2
    dz = zs[1] - zs[0]
    assert dz > 0
    pave = 0.5 * sum(ps)
    Tave = 0.5 * sum(Ts)
    Rave = 0.622/pave * (Rdry/(0.622 - qvap) + Rwat/qvap)
    ρave = pave / Tave / Rave
    # Calculate column number density of water from specific humidity
    H2O = (qvap       # Specific humidity [kg/kg]
            * ρave     # Density of water vapor → [kg/m³]
            / 0.018    # 0.018 kg of water is 1 mol → [mol/m³]
            * avogadro # Number density → [molecules/m³]
            * dz       # Column number density → [molecules/m²]
            * 1.0e-4   # MonoRTM wants cm² → [molecules/cm²]
            )
    # Cloud amout in mm contained in column
    CLW = (qliq   # Specific CLW [kg/kg]
            * ρave # Density of CLW [kg/m³]
            * dz   # Column CLW [kg/m²], corresponds to [mm]
            )
    if CLW == 0: CLW = None
    # Broadening gas amount must be given as column density (see __doc__)
    broadening = mixing_ratio_Ar * dz * (pave*100) / Tave / boltzmann
    # Give species 1 (H2O), 2 (CO2), 7 (O2) and 22 (N2)
    row1 = [H2O, mixing_ratio_CO2, 0., 0., 0., 0., mixing_ratio_O2]
    row2 = [ 0.,               0., 0., 0., 0., 0.,              0.,   0.]
    row3 = [ 0.,               0., 0., 0., 0., 0., mixing_ratio_N2, None]
    # Select Record matching IFORM parameter
    Record211 = Record211_IFORM0 if IFORM == 0 else Record211_IFORM1
    return [Record211(PAVE=pave, TAVE=Tave, ALTZB=zs[0]/1000, PZB=ps[0],
                    TZB=Ts[0], ALTZT=zs[1]/1000, PZT=ps[1], TZT=Ts[1],
                    CLW=CLW), # z in km
            Record212_first(WKL=row1, WBROADL=broadening),
            Record212_other(WKL=row2),
            Record212_other(WKL=row3)
            ]


def from_mwrt_profile(z, p, T, lnq):
    """Output records for building MONORTM_PROF.IN from z, p, T, lnq.
    
    Uses the partioning scheme from mwrt.

    """
    from mwrt.fap import partition_lnq

    out = [Record21()]
    return out
