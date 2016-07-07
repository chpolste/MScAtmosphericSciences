"""Microwave-region radiative transfer in the atmosphere.

A radiative transfer model, stripped to the bare minimum, written only with the
scientific Python stack and with exact differentiation. Provided is code
generation for fast absorption predictors from refractivity models.

Module-wide conventions:
- Temperatures input into refractivity models as reciprocal temperature
  θ = 300/T.
- Individual model components output refractivity N = n - 1, where n is
  refractive index. Use mwrt.fapgen.as_absorption to obtain absorption
  coefficients.
"""

from mwrt.model import MWRTM
from mwrt.fap import FastAbsorptionPredictor
from mwrt.autodiff import DiagVector, Vector
from mwrt.interpolation import LinearInterpolation, atanspace
import mwrt.evaluation as evaluation
import mwrt.refractivity_l93 as liebe93
import mwrt.refractivity_tkc as tkc

# fapgen is not imported by default since it requires sklearn, which is
# unnecessary once the FAP is trained

