# MScAtmosphericSciences

Master's thesis in Atmospheric Sciences.

- __Title__: Bayesian Retrieval of Thermodynamic Atmospheric Profiles from Ground-based Microwave Radiometer Data
- __Submitted__: 2016-08-12 to the faculty of Geo- and Atmospheric Sciences at the University of Innsbruck
- Thesis pdf available [here](MSc-Polster-Bayesian-Retrieval-Thesis.pdf) or at the [library](https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/1471806) of the University of Innsbruck
- Defense presentation pdf available [here](MSc-Polster-Bayesian-Retrieval-Defense.pdf)


## Abstract

Ground-based microwave radiometers are increasingly used in the atmospheric sciences to retrieve vertical temperature, humidity and cloud information. Such information is valuable for boundary layer research and weather forecasting and efforts are undertaken to assimilate microwave radiometer observations into numerical weather prediction models. Multiple methods exist to perform the retrieval, differing in their data requirements, ease of use and flexibility to include measurements from sensors other than the radiometer.

A linear regression and an optimal estimation technique have been implemented as part of this thesis. They are derived from a Bayesian standpoint and important properties of these methods are discussed. Finally, their accuracy is evaluated with data from radiosoundings and radiometer measurements in Innsbruck. Standard deviations of temperature retrievals from an optimal estimation scheme incorporating forecasts from a numerical weather prediction model are found be be less than 1.2 K throughout the troposphere. The least accurate region is located between 1.5 and 3 km above ground level. At these heights the numerical forecasts are not as accurate as in the upper troposphere and the information content of the radiometer has already decreased substantially compared to the lower atmosphere therefore the retrieval scheme struggles to perform well.

Two case studies reveal that the optimal estimation scheme is promising for the retrieval of temperature inversions which are an often studied problem of microwave radiometer retrieval. An experiment shows that the quality of a-priori information, particularly its capability of providing a description of the features that an atmospheric state exhibits, has much influence on the accuracy of retrieved vertical profiles. The a-priori information are therefore a good place to start when trying to improve the retrieval performance.

Also presented in this thesis is a prototype of a numerical radiative transfer model for the microwave region. It is a minimalistic implementation in a high-level programming language and able to calculate linearizations of itself by utilizing automatic differentiation. The model is found to be sufficiently accurate for use in retrieval applications.


## Repository Information

- This repository contains both software and TeX sources, but not the required input data to reproduce the analyses.
- All files in folder [software](software) are licensed under the terms of the MIT license (see [software/LICENSE](software/LICENSE)).

