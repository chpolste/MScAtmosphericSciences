# Software

Data analysis scripts/notebooks and software that will not be developed into standalone tools.

Licensed unter terms of the MIT license (see [LICENSE](LICENSE)).

## Retrieval

- `regression.py`: Bayesian linear regression
- `optimal_estimation.py`: optimal estimation tools with preconfigured setups for HATPRO retrievals
- `run_optest.py` and `run_optest`: perform retrievals based on synthetic data
- `run_optest_hatpro.py` and `run_optest_hatpro`: perform retrievals based on real data
- `Baseline.ipynb`, `OptimalEstimationRetrievals.ipynb` and `RegressionAndMiscRetrievals.ipynb`: Retrieval result visualization and selected experiments.

## Convenience

- `db_tools.py`: helps accessing the main database
- `formulas.py`: constants and conversion functions for meteorological variables

## Visualization

- `Figures.ipynb`: plots for the thesis (some are found in the retrieval notebooks)
- `data_availability.py`: data availability assessment for different sources
- `plots.py`: plot routines for data exploration

## Data Handling

- `db_import.py` and `db_import`: data processing step 1. Takes data from different sources and puts them into a sqlite3 database.
- `data_unifier.py` and `data_unifier`: data processing step 2. Takes data from the main database, runs MWRTM simulations and produces reduced datasets.
- `bufr.py`: a (pretty slow) BUFR reader based on [eccodes](https://software.ecmwf.int/wiki/display/ECC/ecCodes+Home)' `bufr_dump` tool
- `hatpro.py`: HATPRO raw file readers
- `l2e.py`: MeteoSwiss L2E data format parser
- `Priors.ipynb`: generate prior distributions from the unified data
- `DataStats.ipynb`: information about data set

## Radiative Transfer Modelling

- `monortm`: (partial) MonoRTM wrapper (implements only the functionality used in the thesis)
- `mwrt`: custom microwave radiative transfer model with forward mode automatic differentiation (only valid for angles near zenith, no consideration of Earth's curvature or atmospheric refractivity)
- `faps_generator.py`: training of FAPs for HATPRO channels and code generation
- `spectral_line_generator.py`: creates the spectral line database used in `mwrt` from a reference implementation.
- `ModelEvaluation.ipynb`: performance evaluation of MWRTM components

