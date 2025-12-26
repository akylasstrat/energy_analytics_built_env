Technical Evaluation of SMETER Technologies Project Phase 2 Data: HTCs, in-use temperature and energy data, local weather, dwelling information, and occupancy surveys from 30 English dwellings 2019-2021.

---
INTRODUCTION

This dataset was created for the purposes of SMETER Evaluation as part of the Technical Evaluation of SMETER Technologies (TEST) Project led by Prof. David Allinson at Loughborough University.

This Phase 2 dataset used measured co-heating HTCs and asked SMETER Participating Organisations to use in-use data to calculate a corresponding SMETER HTC for each home as part of the SMETER Innovation Competition. The work was funded by the Department of Business, Energy and Industrial Strategy. Refer to the Technical Evaluation of SMETER Technologies (TEST) Project report for further information: (https://hdl.handle.net/2134/19169027).

This dataset is made publicly available to support research and technology development in the area of Smart Meter Enabled Thermal Efficiency Rating (SMETER) Technologies/in-use HTC calculation.

Please download and make use of this dataset in your own research and use the citation below in all publications. We are happy to answer any questions about the dataset and its use. Questions should be sent to Prof. David Allinson and Dr Ben Roberts (d.allinson@lboro.ac.uk; b.m.roberts@lboro.ac.uk).
---
CITATION INFORMATION
Allinson, D., et al. (2022). Technical Evaluation of SMETER Technologies Project Phase 2 Data: HTCs, in-use temperature and energy data, local weather, dwelling information, and occupancy surveys from 30 English dwellings 2019-2021. [Dataset]. UK Data Service. DOI: To be confirmed by UK Data Service.
---
DESCRIPTION OF DATASET
The TEST project (Technical Evaluation of SMETER Technologies) ran from January 2019 until January 2022 and was a collaboration between Loughborough University, Leeds Beckett University, UCL, and Halton Housing. 

The measurement phase of the project (Phase 2) involved 30 dwellings in England and followed Phase 1, which was based on simulated data (https://doi.org/10.5255/UKDA-SN-856292).

This dataset comprises gas and electricity consumption, indoor air temperature, and indoor relative humidity measured in 30 different occupied dwellings. Outdoor weather data measured at a single location within 6 km of all the homes is also available. The dwellings were all located in North West England in two neighbouring towns.

The 30 dwellings have been given a random numerical identifier with a prefix of "HH". The dwellings were vacated by the occupants for one month, during which time a co-heating test, airtightness tests, and other surveys were conducted.

For each house, data were collected for gas consumption (m^3) and electricity consumption (kWh) measured at the meter for the whole dwelling. Indoor air temperature (degC) and relative humidity (%) data were measured in individual rooms. Data were measured at 30 minute intervals. Start dates for the data collection varied and in some cases commenced prior to occupants moving in. Please refer to "SMETER_P2_DwellingInfo.csv" Column H for further information on occupancy dates. All data collection in all houses ended on 31/03/2021 at 23:30. Weather data are available from 26/11/2019 at 00:00 to 02/03/2021 at 08:00.

Weather data comprises wind direction, wind speed, rainfall, air temperature, relative humidity, air pressure, and vertical global solar irradiance. All weather data were measured at 10-minute intervals and the 30-minute mean average taken.

The folders in this dataset:

- "SMETER_P2_Energy_Temp_RH_Weather_30homes.zip"
	- Which contains 30 files (.csv) of electricity consumption, gas consumption, indoor air temperature, relative humidity, and weather data in the format:
		- "HH01.csv" for dwelling HH01
		- "HH02.csv" for dwelling HH02 etc.

The files in this dataset:

- "SMETER_P2_README.txt" (this file).

- "SMETER_P2_QA_weekly_energy_Nov19-Feb21.xlsx" a weekly quality assurance (QA) report of the energy meter data with one week per tab.

- "P2_DwellingInfo.csv" which contains information on the construction, age, type, HTC, and other information about the dwellings.

- "SMETER_P2_Airtightness.csv" which contains information on the airtightness tests done by low-pressure pulse and fan pressursation methods.

- "SMETER_P2_Floorplans.pdf" a floor plan for each of the 30 dwellings.

- "SMETER_P2_WindowArea_Orientation.csv" data on the orientation and size of the windows in each dwelling.

- "SMETER_P2_OccupantInfo.csv" which contains the raw data from the occupant surveys.

- "SMETER_P2_OccupantInfo_SurveyForms.pdf" the two surveys that were delivered.


---
SUMMARY OF DATA COLLECTION
Refer to the Technical Evaluation of SMETER Technologies (TEST) Project report for further information: (https://hdl.handle.net/2134/19169027).

The measured co-heating HTC was compared to a SMETER (in-use) HTC calculated by an independent Quality Assurance (QA) team.
---
DATA STATISTICS
Number of houses: 30
---
FURTHER INFORMATION
For more information about the TEST project please contact Prof. David Allinson (D.Allinson@lboro.ac.uk) or Dr Ben Roberts (B.M.Roberts@lboro.ac.uk).
---
ACKNOWLEDGEMENTS

The TEST Project Technical Assessment Contractor was a consortium of three universities: Loughborough University, Leeds Beckett University, and UCL - and one social housing provider: Halton Housing.

Funding was provided by the Department of Business, Energy and Industrial Strategy via the SMETER Innovation Competition.