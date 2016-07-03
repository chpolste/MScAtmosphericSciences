# Software

Data analysis scripts/notebooks and software that will not be developed into standalone tools.

## Convenience

- `dbtoolbox.py`: wrapper for the main database
- `formulas.py`: constants and conversion functions for meteorological variables

## Visualization

- `Figures.ipynb`: plots for the thesis
- `data_availability.py`: data availability assessment for different sources
- `plots.py`: plot routines for data exploration

## Data Handling

- `bufr.py`: a (pretty slow) BUFR reader based on [eccodes](https://software.ecmwf.int/wiki/display/ECC/ecCodes+Home)' `bufr_dump` tool
- `db_create`: bash script that assembles the main database (main use is for multi-process parsing of bufr files, which otherwise takes forever)
- `db_import.py`: takes data from different sources and puts them into a sqlite3 database
- `hatpro.py`: HATPRO raw file readers
- `l2e.py`: MeteoSwiss L2E data format parser

## Radiative Transfer Modelling

- `mwrt`: Custom microwave radiative transfer model (work in progress)