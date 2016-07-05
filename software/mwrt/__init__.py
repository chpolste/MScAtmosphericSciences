"""Microwave-region radiative transfer in the atmosphere.


Module-wide conventions
-----------------------

Temperatures input as reciprocal temperature θ = 300/T

Individual model components output refractivity N = n - 1, where n is
refractive index.

To obtain absorption coefficient: 4*pi*ν/c * Im(N)
"""

import mwrt.refractivity_l93 as liebe93
import mwrt.refractivity_tkc as tkc
from mwrt.fap import FastAbsorptionPredictor
from mwrt.fapgen import CloudFAPGenerator, GasFAPGenerator, as_absorption, absorption_model, FAPGenerator
from mwrt.model import MWRTM

