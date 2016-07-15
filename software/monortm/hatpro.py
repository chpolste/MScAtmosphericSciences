"""MonoRTM routines and configuration for retrievals with HATPRO."""

from monortm.records import Record12, Record13, Record131, Record132, Record14


Mhz2icm = 1 / 29.9792458 # = 1.0e9 / speed of light in cm/s

# Molecular mixing ratios
# Source: https://en.wikipedia.org/wiki/Atmosphere_of_Earth#Composition
mixing_ratio_N2 = 0.78084
mixing_ratio_O2 = 0.20946
mixing_ratio_Ar = 0.009340
mixing_ratio_CO2 = 0.000397


config = [
        Record11(TOKEN="$", CXID="created for HATPRO simulation"),
        Record12(IHIRAC=1, ICTNM=1, IEMIT=1, IPLOT=1, IATM=0, IXSECT=0, IBRD=1),
        Record13(V1=-99., V2=-99., DVSET=None, NMOL_SCAL=None),
        Record131(NWN=14),
        Record132(WN=22240*MHz2icm),
        Record132(WN=23040*MHz2icm),
        Record132(WN=23840*MHz2icm),
        Record132(WN=25440*MHz2icm),
        Record132(WN=26240*MHz2icm),
        Record132(WN=27840*MHz2icm),
        Record132(WN=31400*MHz2icm),
        Record132(WN=51260*MHz2icm),
        Record132(WN=52280*MHz2icm),
        Record132(WN=53860*MHz2icm),
        Record132(WN=54940*MHz2icm),
        Record132(WN=56660*MHz2icm),
        Record132(WN=57300*MHz2icm),
        Record132(WN=58000*MHz2icm),
        Record11(TOKEN="%")
        ]


def from_mwrt_profile(z, p, T, lnq):
    """Output records for building MONORTM_PROF.IN from z, p, T, lnq.
    
    Uses the partioning scheme from mwrt.

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
    from mwrt.fap import partition_lnq

    out = [Record21()]
    for layer in layers:
        out.extend([
                Record211(),
                Record212_first(),
                Record212_other(),
                Record212_other(),
                ])
    return out
