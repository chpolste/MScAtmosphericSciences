"""Microwave-region radiative transfer in the atmosphere.


Module-wide conventions
-----------------------

Temperatures input as reciprocal temperature θ = 300/T

Individual model components output refractivity N = n - 1, where n is
refractive index.

To obtain absorption coefficient: 4*pi*ν/c * Im(N)
"""

import .refractivity_l93 as liebe93
import .refractivity_tkc as tkc


