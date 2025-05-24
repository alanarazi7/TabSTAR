from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: rmftsa_ladata
====
Examples: 508
====
URL: https://www.openml.org/search?type=data&id=666
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Data Sets for 'Regression Models for Time Series Analysis' by
B. Kedem and K. Fokianos, Wiley 2002. Submitted by Kostas
Fokianos (fokianos@ucy.ac.cy) [8/Nov/02] (176k)

Note: - attribute names were generated manually
- information about data taken from here:
http://lib.stat.cmu.edu/datasets/

File: ../data/rmftsa/ladata

LA Pollution-Mortality Study:
1970-1979, 508 observations,  6-day spacing. Weekly FILTERED data.
The data were lowpass filtered, filtering out frequencies above 0.1
cycles per day.
Mortality:          (1) Mrt: Total Mortality
(2) Rsp: Respiratory Mortality
(3) Crd: Cardiovascular Mortality
Weather:            (4) Tmp: Temperature
(5) Hum: Relative Humidity
Pollution:          (6) Crb: Carbon Monoxide
(7) Slf: Sulfur Dioxideglm.LAshumway
(8) Nit: Nitrogen Dioxide
(9) Hdr: Hydrocarbons
(10) Ozn: Ozone
(11) Par: Particulates


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific
====
Target Variable: Respiratory_Mortality (numeric, 365 distinct): ['6.56', '7.89', '6.39', '7.8', '9.08', '6.19', '6.94', '6.53', '5.23', '6.72']
====
Features:

Total_Mortality (numeric, 483 distinct): ['151.92', '165.24', '165.98', '180.78', '166.52', '158.66', '165.42', '160.43', '167.55', '206.97']
Cardiovascular_Mortality (numeric, 478 distinct): ['87.33', '97.4', '85.51', '89.35', '81.17', '81.53', '87.02', '86.83', '94.19', '90.48']
Temperature (numeric, 473 distinct): ['63.95', '78.61', '68.35', '83.13', '64.28', '80.23', '82.16', '77.99', '71.89', '68.55']
Relative_Humidity (numeric, 478 distinct): ['57.97', '52.8', '69.7', '61.68', '55.28', '65.67', '66.0', '55.55', '65.75', '68.02']
Carbon_Monoxide (numeric, 409 distinct): ['5.14', '4.84', '3.88', '5.41', '4.88', '4.86', '4.66', '7.42', '10.19', '10.55']
Sulfur_Dioxideglm.LAshumway (numeric, 282 distinct): ['2.43', '2.76', '3.13', '1.79', '2.0', '1.86', '1.85', '2.78', '2.24', '2.53']
Nitrogen_Dioxide (numeric, 430 distinct): ['10.0', '8.74', '9.7', '8.64', '11.35', '9.82', '8.6', '7.87', '10.05', '7.7']
Hydrocarbons (numeric, 479 distinct): ['59.88', '38.63', '39.92', '43.54', '64.06', '65.56', '52.93', '45.12', '60.13', '53.14']
Ozone (numeric, 432 distinct): ['9.99', '5.71', '3.85', '4.64', '3.12', '9.76', '3.07', '2.12', '6.91', '8.86']
Particulates (numeric, 494 distinct): ['32.26', '38.51', '39.16', '42.62', '57.53', '48.69', '37.26', '54.2', '70.13', '33.96']
'''

CONTEXT = "Pollution Mortality Study in Los Angeles"
TARGET = CuratedTarget(raw_name="Respiratory_Mortality", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []