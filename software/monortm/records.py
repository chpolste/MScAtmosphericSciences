from monortm.meta import Record


# Record group 1

class Record11(Record):
    """"""

    TOKEN = ("A1", " ", str,
            """Put the $ indicating a new section here.""")

    CXID = ("A79", None, str,
            """User identification.""")


class Record12(Record):
    """"""

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
            """Flag for species specific broadening: 1 = yes, 0 = no.""")


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


# Record group 2
# These records are applicable only if LBLATM not selected
# (IATM = 0). These records should be in a separate file: MONORTM_PROF.IN.
# Layer input (molecules only).


class Record21(Record):
    """"""
    
    IFORM = ("1X,I1", 1, int,
            """Column amount format flag.""")

    NLAYRS = ("I3", None, range(1, 201),
            """Number of layers (maximum of 200).""")

    NMOL = ("I5", None, int,
            """Number of highest molecule number used (default 7, maximum of 35).""")

    SECNTO = ("F10.2", None, float,
            """+1 = looking up, -1 = looking down.""")

    ZH1 = ("20X,F8.2", None, float,
            """Observer altitude.""")

    ZH2 = ("4X,F8.2", None, float,
            """End point altitude.""")

    ANGLE = ("5X,F8.3", None, float,
            """Zenith angle at H1 (degrees). For nadir looking up: ANGLE = 0,
            for nadir looking down: ANGLE = 180.""")


class Record211(Record):
    """ALTZB, PZB, TZB are only required for the first layer. MONORTM assumes
    that these quantities are equal to the top of the previous layer for other
    layers."""

    PAVE = ("F10.4", None, float,
            """Average pressure of layer (millibars).""")

    TAVE = ("F10.4", None, float,
            """Average temperature of layer (K).""")

    # SECNTK, ITYL, IPATH are not used in MONORTM. Format would be F10.4,A3,I2,1X, therefore
    # a gap of 16 characters is added to ALTZB:
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

    CLW = ("E15.7", None, float,
            """Cloud Liquid Amount in mm present in the layer.""")


class Record212(Record):
    """"""

    WKL = ("E10.3", None, (float,)*7,
            """Column densities or mixing ratios for 7 molecular species (molecules/cm²).""")

    WBROADL = ("E10.3", None, float,
            """Column density for broadening gases (molecules/cm²).""")

