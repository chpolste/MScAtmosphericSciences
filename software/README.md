# Software

Data analysis scripts/notebooks and software that will not be developed into standalone tools.

## Retrieval

- `regression.py`: Bayesian linear regression
- `optimal_estimation.py`: optimal estimation tools (work in progress)

## Convenience

- `db_tools.py`: helps accessing the main database
- `formulas.py`: constants and conversion functions for meteorological variables

## Visualization

- `Figures.ipynb`: plots for the thesis
- `data_availability.py`: data availability assessment for different sources
- `plots.py`: plot routines for data exploration

## Data Handling

- `bufr.py`: a (pretty slow) BUFR reader based on [eccodes](https://software.ecmwf.int/wiki/display/ECC/ecCodes+Home)' `bufr_dump` tool
- `data_unifier`: bash script that creates csv files from database containing reduced datasets for quick access
- `data_unifier.py`: takes data from the main database and produces reduced datasets
- `db_import`: bash script that assembles the main database (main use is for multi-process parsing of bufr files, which otherwise takes forever)
- `db_import.py`: takes data from different sources and puts them into a sqlite3 database
- `hatpro.py`: HATPRO raw file readers
- `l2e.py`: MeteoSwiss L2E data format parser

## Radiative Transfer Modelling

- `monortm`: (partial) MonoRTM wrapper
- `mwrt`: custom microwave radiative transfer model with forward mode automatic differentiation (only valid for angles near zenith, no consideration of Earth's curvature or atmospheric refractivity)
- `faps_generator.py`: training of FAPs for HATPRO channels and code generation
- `spectral_line_generator.py`: creates the spectral line database used in `mwrt` from a reference implementation.
