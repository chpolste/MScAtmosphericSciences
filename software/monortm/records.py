"""MonoRTM record definitions."""

from monortm.meta import RecordMeta


class Record(metaclass=RecordMeta):
    """Base class for records applying metaclass and adding common methods."""
    
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self._values = {}
        return self

    def __init__(self, **kwargs):
        """Instanciate a new record."""
        for key, val in kwargs.items():
            if key in self._order:
                # The default values are already set by the metaclass before
                # __init__ is called.
                setattr(self, key, val)
            else:
                err = "Element {} not in {}.".format(key, type(self).__name__)
                raise ValueError(err)

    def __str__(self):
        """Return with formatting applied."""
        return "".join(getattr(self, key) for key in self._order)


class Record11(Record):

    TOKEN = ("A1", " ", str,
            """Put the $ indicating a new section here.""")

    CXID = ("A79", None, str,
            """User identification.""")


class Record12(Record):

    IHIRAC = ("4X,I1", 1, int,
            """Should be one: Voigt profile.""")

    ICNTNM = ("9X,I1", 1, {0, 1, 2, 3, 4, 5, 6},
            """Flag for continuum. 0 = no, 1 = all, ...""")

    IEMIT = ("9X,I1", 1, int,
            """Will always be set internally to 1 to calculate radiances.""")

    IPLOT = ("14X,I1", 1, {0, 1},
            """Flag for calculation of brightness temperatures as well as radiances.
            0 = radiance only, 1 = radiance and brightness temperatures.""")

    IATM = ("9X,I1", None, {0, 1},
            """Flag for LBLATM: 1 = yes, 0 = no (then additional input file must be given).""")

    IOD = ("14X,I1", 0, int,
            """Write out layer optical depths? 0 = no, 1 = yes.""")

    IXSECT = ("4X,I1", 0, {0, 1},
            """Cross section flag.""")

    ISPD = ("16X,I4", 0, int,
            """???""")

    IBRD = ("I4", None, {0, 1},
            """Flag for species specific broadening: 1 = yes, 0 = no. Setting
            IBRD to zero turns off the species specific broadening,
            significantly shortening the processing time. This is a reasonable
            option for users working in scenarios where these effects are
            small.""")


class Record13(Record):
    """Used to compute simulations for a set of equally-spaced wavenumbers. For
    simulations for a number of non-equally-spaced wavenumbers, set V1 and/or
    V2 to any negative value. MONORTM then expects records 1.3.1 and 1.3.2."""

    V1 = ("E10.3", None, float,
            """Beginning wavenumber value for the calculation. If negative, see
            record 1.3.1.""")

    V2 = ("E10.3", None, float,
            """Ending wavenumber value for the calculation. If negative, see
            record 1.3.1.""")

    DVSET = ("10X,E10.3", None, float,
            """Stepsize of the monochromatic calculation by MONORTM.""")

    NMOL_SCAL = ("63X,I2", 0, range(0, 39),
            """Enables the scaling of the atmospheric profile for selected
            species. NMOL_SCAL is the highest molecule number for which scaling
            will be applied. See Record(s) 1.3.a/1.3.b.n. Only works for IATM=1.""")


class Record131(Record):
    """Required only if V1 < 0 or V2 < 0."""

    NWN = ("I8", None, int,
            """Number of wavenumbers to be read in record 1.3.2.""")


class Record132(Record):
    """Required only if V1 < 0 or V2 < 0. Repeat NWN times."""

    WN = ("E19.7", None, float,
            """Wavenumber to be processed by MONORTM. cm^-1 without 2pi.""")


class Record14(Record):
    """Temperature and emissivity parameters for boundary at H2 (end of path)."""

    TBOUND = ("E10.3", None, float,
            """Temperature of boundary (K). Internally set to 2.75 K for
            downwelling radiance.""")

    SREMIS = ("E10.3", None, (float, float, float),
            """Frequency dependent boundary emissivity coefficients:

                EMISSIVITY = SREMIS[0] + SREMIS[1]*V + SREMIS[2]*(V**2).

            A negative value for element 0 allows direct input of boundary
            emissivities from file 'EMISSION'""")

    SRREFL = ("E10.3", None, (float, float, float),
            """Frequency dependent boundary reflectivity coefficients:

                REFLECTIVITY = SRREFL[0] + SRREFL[1]*V + SRREFL[2]*(V**2).

            A negative value for element 0 allows direct input of boundary
            reflectivities from file 'REFLECTION'""")


# Record group 2 is applicable only if LBLATM not selected (IATM = 0). These
# records should be in a separate file: MONORTM_PROF.IN.

class Record21(Record):
    
    IFORM = ("1X,I1", 1, int,
            """Column amount format flag. Use corresponding versions of
            records 2.1.1 and 2.1.2.""")

    NLAYRS = ("I3", None, range(1, 201),
            """Number of layers (maximum of 200).""")

    NMOL = ("I5", None, int,
            """Number of highest molecule number used (default 7, maximum of 35).""")

    # Instructions say F10.2 but source code and example files have F10.6.
    SECNTO = ("F10.6", None, float,
            """+1 = looking up, -1 = looking down.""")

    H1 = ("20X,F8.2", None, float,
            """Observer altitude.""")

    H2 = ("4X,F8.2", None, float,
            """End point altitude.""")

    ANGLE = ("5X,F8.3", None, float,
            """Zenith angle at H1 (degrees). For nadir looking up: ANGLE = 0,
            for nadir looking down: ANGLE = 180.""")

    LEN = ("5X,I2", None, int,
            """Apparently not used by MonoRTM. Is 0 in example profiles.""")


class Record211_IFORM0(Record):
    """Use this version of record 2.1.1 if IFORM = 0 in record 2.1.
    
    ALTZB, PZB, TZB are only required for the first layer. MONORTM assumes
    that these quantities are equal to the top of the previous layer for other
    layers."""

    PAVE = ("F10.4", None, float,
            """Average pressure of layer (millibars = hPa).""")

    TAVE = ("F10.4", None, float,
            """Average temperature of layer (K).""")

    # SECNTK, ITYL, IPATH are not used in MONORTM. Format would be
    # F10.4,A3,I2,1X, therefore a gap of 16 characters is added to ALTZB
    ALTZB = ("16X,F7.2", None, float,
            """Altitude for bottom of current layer.""")

    PZB = ("F8.3", None, float,
            """Pressure at ALTZB.""")

    TZB = ("F7.2", None, float,
            """Temperature at ALTZB. Used by MONORTM for Planck function.""")

    ALTZT = ("F7.2", None, float,
            """Altitude for top of current layer.""")

    PZT = ("F8.3", None, float,
            """Pressure at ALTZT.""")

    TZT = ("F7.2", None, float,
            """Temperature at ALTZT.""")

    # Instructions say E15.7 but monortm.f90 has F7.2. In tests up to 4 decimal
    # places were read so this is what is used for maximum accuracy
    CLW = ("F7.4", None, float,
            """Cloud Liquid Amount in mm present in the layer.""")


class Record211_IFORM1(Record):
    """Use this version of record 2.1.1 if IFORM = 1 in record 2.1.
    
    ALTZB, PZB, TZB are only required for the first layer. MONORTM assumes
    that these quantities are equal to the top of the previous layer for other
    layers."""

    PAVE = ("E15.7", None, float,
            """Average pressure of layer (millibars = hPa).""")

    TAVE = ("F10.4", None, float,
            """Average temperature of layer (K).""")

    # SECNTK, ITYL, IPATH are not used in MONORTM. Format would be
    # F10.4,A3,I2,1X, therefore a gap of 16 characters is added to ALTZB
    ALTZB = ("16X,F7.2", None, float,
            """Altitude for bottom of current layer (information only).""")

    PZB = ("F8.3", None, float,
            """Pressure at ALTZB (information only).""")

    TZB = ("F7.2", None, float,
            """Pressure at ALTZB. Used by MONORTM for Planck Function Calculation.""")

    ALTZT = ("F7.2", None, float,
            """Altitude for top of current layer (information only).""")

    PZT = ("F8.3", None, float,
            """Pressure at ALTZT (information only).""")

    TZT = ("F7.2", None, float,
            """Temperature at ALTZT.""")

    # Instructions say E15.7 but monortm.f90 has F7.2. In tests up to 4 decimal
    # places were read so this is what is used for maximum accuracy
    CLW = ("F7.4", None, float,
            """Cloud Liquid Amount in mm present in the layer.""")


class Record212_first(Record):
    """First line of constituent specifications containing broadening
    gases in last element (this element always has to contain column density).
    
    The instructions say there are two versions of this record depending of
    IFORM in record 2.1 but in the source code the version with E15.7 is read
    in both cases.
    """

    WKL = ("E15.7", None, (float,)*7,
            """Column densities or mixing ratios for 7 molecular species.""")

    WBROADL = ("E15.7", None, float,
            """Column density for broadening gases (molecules/cm²).""")


class Record212_other(Record):
    """Repeat this line after the first until all consituents are specified.

    The instructions say there are two versions of this record depending of
    IFORM in record 2.1 but in the source code the version with E15.7 is read
    in both cases..."""

    WKL = ("E15.7", None, (float,)*8,
            """Column densities or mixing ratios for 8 molecular species.""")

