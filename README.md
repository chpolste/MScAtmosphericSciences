# MScAtmosphericSciences

Master's thesis in Atmospheric Sciences.

- __Title__
  Bayesian Retrieval of Thermodynamic Atmospheric Profiles from Ground-based Microwave Radiometer Data
- __Submitted__
  2016-08-12 to the faculty of Geo- and Atmospheric Sciences at the University of Innsbruck


## Abstract

Ground-based microwave radiometers are increasingly used to retrieve vertical temperature, humidity and cloud information of the atmosphere. Such information is valuable for boundary layer research, weather forecasting and experiments have been undertaken to assimilate radiometer observations into numerical weather prediction models. Multiple methods exist to perform the retrieval, differing in their data requirements, ease of use and flexibility to include measurements from sensors other than the radiometer.

A linear regression and an optimal estimation technique have been derived and implemented in this thesis. Important properties of these methods are discussed and their accuracy is evaluated with data from radiosoundings and radiometer measurements in Innsbruck. Standard deviations of temperature retrievals from an optimal estimation scheme integrating forecasts from a numerical weather prediction model are found be be less than 1.2 K throughout the troposphere relative to reference measurements from radiosoundings. The least accurate region is located between 1.5 and 3 km above ground level. There the numerical forecasts are not as accurate as in the upper troposphere and the information content of the radiometer has already decreased substantially compared to the lower atmosphere.

In two case studies it is found that the optimal estimation scheme is promising for the retrieval of temperature inversions which have been an often studied problem of microwave radiometer retrieval. The quality of a-priori information, particularly its capability of providing a first guess of the features that an atmospheric state exhibits is, found to be a major influence on the retrieved vertical profiles.

Also presented in this thesis is a prototype of a numerical radiative transfer model for the microwave region. It is a minimalistic implementation in a high-level programming language and able to calculate linearizations of itself by automatic differentiation. The model is found to be sufficiently accurate for use in retrieval applications.


## Repository Information

- This is a monorepo for code and TeX source.
- The code is documented in most places but I did not write automated tests for the software (sadly).
- ...
