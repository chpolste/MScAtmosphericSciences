# Software

Data analysis scripts/notebooks and software that will not be developed into standalone tools.


## raso

Convenience container for vertical profiles of the atmosphere.

- A `Profile` class wrapping a structured numpy array that contains the numerical data and stores associated metadata
- Quick generation of decent plots
- Automatic derivation of quantities that can be calculated from the data contained in the `Profile` (e.g. potential temperature from pressure and temperature)
- [doctab](https://github.com/chpolste/doctab) integration for saving and querying `Profile` instances
- A (crude) BUFR reader based on [eccodes](https://software.ecmwf.int/wiki/display/ECC/ecCodes+Home)' `bufr_dump` tool
