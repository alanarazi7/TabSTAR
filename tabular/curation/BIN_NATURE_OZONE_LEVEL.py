from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ozone-level-8hr
====
Examples: 2534
====
URL: https://www.openml.org/search?type=data&id=1487
====

Paper: https://ieeexplore.ieee.org/document/4053100

Description: **Author**: Kun Zhang, Wei Fan, XiaoJing Yuan

**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/ozone+level+detection)

**Please cite**:   

Forecasting skewed biased stochastic ozone days: analyses, solutions and beyond, Knowledge and Information Systems, Vol. 14, No. 3, 2008. 


1 . Abstract: 
Two ground ozone level data sets are included in this collection. One is the eight hour peak set (eighthr.data), the other is the one hour peak set (onehr.data). Those data were collected from 1998 to 2004 at the Houston, Galveston and Brazoria area.

2. Source:

Kun Zhang, zhang.kun05 '@' gmail.com, Department of Computer Science, Xavier University of Lousiana 
Wei Fan, wei.fan '@' gmail.com, IBM T.J.Watson Research 
XiaoJing Yuan, xyuan '@' uh.edu, Engineering Technology Department, College of Technology, University of Houston 


3. Data Set Information:

All the attribute start with T means the temperature measured at different time throughout the day; and those starts with WS indicate the wind speed at various time. 

WSR_PK: continuous. peek wind speed -- resultant (meaning average of wind vector) 
WSR_AV: continuous. average wind speed 
T_PK: continuous. Peak T 
T_AV: continuous. Average T 
T85: continuous. T at 850 hpa level (or about 1500 m height) 
RH85: continuous. Relative Humidity at 850 hpa 
U85: continuous. (U wind - east-west direction wind at 850 hpa) 
V85: continuous. V wind - N-S direction wind at 850 
HT85: continuous. Geopotential height at 850 hpa, it is about the same as height at low altitude 
T70: continuous. T at 700 hpa level (roughly 3100 m height) 

RH70: continuous. 
U70: continuous. 
V70: continuous. 
HT70: continuous. 

T50: continuous. T at 500 hpa level (roughly at 5500 m height) 

RH50: continuous. 
U50: continuous. 
V50: continuous. 
HT50: continuous. 

KI: continuous. K-Index [Web Link] 
TT: continuous. T-Totals [Web Link] 
SLP: continuous. Sea level pressure 
SLP_: continuous. SLP change from previous day 

Precp: continuous. -- precipitation


4. Attribute Information:

The following are specifications for several most important attributes that are highly valued by Texas Commission on Environmental Quality (TCEQ). More details can be found in the two relevant papers. 

O 3 - Local ozone peak prediction 
Upwind - Upwind ozone background level 
EmFactor - Precursor emissions related factor 
Tmax - Maximum temperature in degrees F 
Tb - Base temperature where net ozone production begins (50 F) 
SRd - Solar radiation total for the day 
WSa - Wind speed near sunrise (using 09-12 UTC forecast mode) 
WSp - Wind speed mid-day (using 15-21 UTC forecast mode) 


5. Relevant Papers:

Forecasting skewed biased stochastic ozone days: analyses, solutions and beyond, Knowledge and Information Systems, Vol. 14, No. 3, 2008. 

It Discusses details about the dataset, its use as well as various experiments (both cross-validation and streaming) using many state-of-the-art methods. 
A shorter version of the paper (does not contain some detailed experiments as the journal paper above) is in: 
Forecasting Skewed Biased Stochastic Ozone Days: Analyses and Solutions. ICDM 2006: 753-764
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 69 distinct): ['1.6402', '0.4', '0.8', '0.3', '1.3', '1.7', '1.1', '0.2', '0.5', '1.0']
V2 (numeric, 71 distinct): ['1.5864', '0.4', '0.8', '0.3', '0.2', '1.3', '0.6', '0.7', '0.1', '0.9']
V3 (numeric, 66 distinct): ['1.5456', '0.4', '0.8', '0.3', '1.3', '0.1', '0.2', '0.9', '0.7', '1.0']
V4 (numeric, 67 distinct): ['1.5264', '0.4', '0.8', '0.3', '1.3', '0.2', '0.6', '0.1', '1.1', '0.5']
V5 (numeric, 65 distinct): ['1.5226', '0.4', '0.8', '0.2', '0.3', '1.3', '1.7', '0.5', '0.6', '1.0']
V6 (numeric, 64 distinct): ['1.5424', '0.4', '0.8', '1.3', '1.4', '0.7', '0.2', '1.0', '0.6', '0.5']
V7 (numeric, 67 distinct): ['1.6379', '1.3', '0.8', '1.2', '0.4', '1.7', '1.4', '0.3', '0.5', '1.0']
V8 (numeric, 68 distinct): ['2.0471', '1.7', '1.3', '2.1', '1.5', '1.4', '0.8', '2.5', '1.6', '1.9']
V9 (numeric, 70 distinct): ['2.539', '2.1', '3.0', '2.5', '1.3', '2.7', '2.4', '3.4', '1.7', '2.0']
V10 (numeric, 71 distinct): ['2.8477', '3.4', '2.5', '3.0', '3.8', '2.1', '2.7', '1.7', '2.8', '3.2']
V11 (numeric, 77 distinct): ['2.9698', '2.5', '2.1', '3.8', '3.4', '3.0', '2.6', '1.7', '2.8', '2.2']
V12 (numeric, 78 distinct): ['3.0164', '2.1', '2.5', '3.0', '3.8', '3.4', '3.6', '2.6', '2.3', '1.7']
V13 (numeric, 78 distinct): ['3.0441', '3.4', '2.1', '2.5', '3.0', '3.8', '2.8', '2.2', '3.3', '2.0']
V14 (numeric, 79 distinct): ['3.1067', '3.0', '2.1', '2.5', '3.4', '3.8', '2.3', '4.2', '3.5', '1.9']
V15 (numeric, 78 distinct): ['3.1777', '3.0', '3.4', '2.5', '3.8', '4.2', '2.1', '2.4', '1.7', '4.6']
V16 (numeric, 79 distinct): ['3.231', '3.4', '3.8', '3.0', '2.5', '3.7', '4.2', '2.1', '2.0', '4.0']
V17 (numeric, 73 distinct): ['3.1928', '3.4', '2.5', '4.2', '3.0', '3.8', '2.1', '3.7', '3.6', '2.6']
V18 (numeric, 74 distinct): ['2.9346', '2.1', '3.4', '3.8', '2.5', '3.0', '2.6', '3.2', '2.4', '1.7']
V19 (numeric, 71 distinct): ['2.5613', '2.5', '2.1', '1.3', '1.7', '3.4', '3.0', '2.4', '2.2', '3.8']
V20 (numeric, 66 distinct): ['2.2856', '1.7', '2.5', '2.1', '3.0', '1.3', '2.0', '2.7', '3.4', '1.6']
V21 (numeric, 69 distinct): ['2.0896', '1.7', '1.3', '2.1', '3.0', '2.5', '1.8', '1.4', '1.6', '1.5']
V22 (numeric, 70 distinct): ['1.9384', '1.3', '1.7', '1.6', '2.1', '1.5', '1.0', '1.4', '1.2', '1.9']
V23 (numeric, 69 distinct): ['1.8038', '1.3', '1.7', '0.8', '1.2', '0.4', '2.1', '1.0', '0.9', '1.1']
V24 (numeric, 66 distinct): ['1.7085', '1.3', '0.4', '0.8', '1.1', '0.9', '0.3', '1.0', '1.6', '0.2']
V25 (numeric, 75 distinct): ['4.1721', '3.4', '3.8', '4.2', '4.6', '4.1', '4.4', '3.6', '4.3', '3.9']
V26 (numeric, 56 distinct): ['2.3149', '1.5', '1.6', '1.9', '1.7', '2.0', '2.1', '1.8', '1.4', '2.4']
V27 (numeric, 283 distinct): ['18.6493', '26.2', '25.2', '26.4', '25.1', '26.6', '25.8', '26.7', '26.1', '23.3']
V28 (numeric, 285 distinct): ['18.3479', '26.4', '24.9', '25.6', '26.2', '26.1', '25.9', '22.6', '24.1', '23.7']
V29 (numeric, 288 distinct): ['18.0609', '26.1', '25.9', '21.8', '24.9', '24.8', '25.3', '25.6', '25.8', '25.4']
V30 (numeric, 284 distinct): ['17.8213', '24.7', '25.7', '25.2', '26.1', '25.4', '25.9', '26.4', '25.8', '25.1']
V31 (numeric, 284 distinct): ['17.6113', '25.2', '26.1', '25.7', '25.8', '24.3', '24.8', '24.2', '25.1', '25.3']
V32 (numeric, 292 distinct): ['17.4755', '24.1', '25.2', '24.7', '24.9', '25.8', '25.6', '24.2', '25.1', '23.4']
V33 (numeric, 296 distinct): ['17.5894', '25.1', '26.1', '25.7', '25.4', '24.9', '25.9', '25.2', '25.6', '24.2']
V34 (numeric, 312 distinct): ['18.4178', '26.6', '27.7', '27.1', '26.8', '20.1', '26.4', '27.3', '25.4', '25.6']
V35 (numeric, 314 distinct): ['19.7788', '28.6', '28.3', '27.9', '27.8', '28.8', '28.4', '28.1', '27.6', '29.2']
V36 (numeric, 315 distinct): ['21.2169', '29.4', '29.8', '29.9', '30.2', '28.1', '28.9', '29.2', '29.6', '29.3']
V37 (numeric, 328 distinct): ['22.4625', '30.3', '31.1', '30.2', '30.9', '25.6', '30.7', '30.6', '30.4', '30.8']
V38 (numeric, 331 distinct): ['23.3936', '31.1', '31.3', '31.4', '32.3', '31.2', '24.6', '32.2', '32.1', '24.7']
V39 (numeric, 335 distinct): ['24.0252', '31.8', '33.2', '26.6', '30.8', '29.1', '32.2', '32.7', '31.2', '32.9']
V40 (numeric, 336 distinct): ['24.4331', '30.2', '32.2', '30.1', '24.8', '24.7', '26.3', '29.2', '30.4', '27.4']
V41 (numeric, 336 distinct): ['24.7051', '26.1', '29.6', '24.4', '26.7', '25.2', '30.4', '24.1', '26.3', '23.6']
V42 (numeric, 340 distinct): ['24.7202', '24.1', '30.6', '24.8', '30.2', '32.3', '27.8', '23.4', '31.3', '29.3']
V43 (numeric, 338 distinct): ['24.3975', '26.6', '24.8', '24.4', '25.4', '30.7', '25.2', '25.6', '29.3', '25.3']
V44 (numeric, 330 distinct): ['23.6321', '28.9', '24.1', '30.6', '30.9', '24.3', '24.6', '28.7', '22.1', '31.7']
V45 (numeric, 322 distinct): ['22.5097', '24.2', '29.3', '25.3', '27.4', '23.4', '26.9', '30.7', '26.6', '29.1']
V46 (numeric, 307 distinct): ['21.4257', '24.4', '25.8', '26.3', '29.4', '26.1', '22.4', '25.1', '28.7', '29.1']
V47 (numeric, 303 distinct): ['20.6154', '28.6', '20.7', '28.1', '28.7', '24.1', '27.9', '21.8', '28.8', '26.1']
V48 (numeric, 295 distinct): ['20.0316', '26.8', '20.1', '25.8', '27.3', '25.9', '24.3', '28.1', '26.7', '23.6']
V49 (numeric, 288 distinct): ['19.5026', '27.1', '26.1', '26.9', '22.7', '24.2', '26.6', '24.3', '27.6', '25.9']
V50 (numeric, 285 distinct): ['19.0623', '27.2', '26.6', '25.6', '26.4', '26.3', '27.3', '26.7', '25.9', '26.2']
V51 (numeric, 331 distinct): ['25.5783', '30.8', '31.7', '31.8', '33.6', '26.1', '30.9', '31.1', '29.6', '26.7']
V52 (numeric, 297 distinct): ['20.8405', '28.4', '28.7', '27.2', '28.5', '26.3', '28.8', '26.9', '27.3', '25.6']
V53 (numeric, 252 distinct): ['13.5753', '18.1', '16.8', '17.1', '17.5', '18.0', '17.4', '18.3', '17.8', '15.6']
V54 (numeric, 101 distinct): ['0.5773', '0.8', '0.79', '0.76', '0.86', '0.64', '0.66', '0.83', '0.77', '0.81']
V55 (numeric, 1289 distinct): ['2.1365', '0.0', '0.29', '-2.73', '1.93', '1.9', '0.26', '1.06', '4.63', '3.47']
V56 (numeric, 1462 distinct): ['1.6625', '-1.76', '0.42', '0.27', '-0.15', '2.22', '2.54', '-2.81', '2.6', '-4.23']
V57 (numeric, 369 distinct): ['1531.4943', '1552.0', '1517.0', '1520.0', '1532.0', '1534.5', '1540.0', '1564.0', '1565.5', '1538.5']
V58 (numeric, 246 distinct): ['5.9311', '9.1', '7.3', '9.0', '8.8', '8.9', '8.1', '8.5', '8.2', '8.4']
V59 (numeric, 101 distinct): ['0.4064', '0.02', '0.03', '0.07', '0.09', '0.2', '0.28', '0.12', '0.53', '0.15']
V60 (numeric, 1538 distinct): ['5.4596', '0.53', '8.23', '11.82', '3.27', '0.74', '3.26', '1.47', '3.38', '6.84']
V61 (numeric, 1430 distinct): ['0.994', '0.0', '-0.8', '4.88', '-1.14', '0.18', '1.99', '2.05', '-2.28', '-0.66']
V62 (numeric, 442 distinct): ['3145.4205', '3150.5', '3165.5', '3166.5', '3160.5', '3170.0', '3157.0', '3194.0', '3167.0', '3163.5']
V63 (numeric, 187 distinct): ['-10.5114', '-6.8', '-8.2', '-7.6', '-6.6', '-6.7', '-7.9', '-7.0', '-7.5', '-7.3']
V64 (numeric, 101 distinct): ['0.3047', '0.03', '0.04', '0.02', '0.05', '0.06', '0.09', '0.15', '0.1', '0.08']
V65 (numeric, 1688 distinct): ['9.8724', '7.9', '8.45', '7.85', '10.91', '12.37', '14.88', '7.07', '7.31', '4.21']
V66 (numeric, 1510 distinct): ['0.8301', '0.0', '-2.46', '0.49', '-0.27', '-0.94', '1.73', '1.4', '-0.64', '2.39']
V67 (numeric, 86 distinct): ['5818.8212', '5885.0', '5895.0', '5900.0', '5880.0', '5865.0', '5855.0', '5860.0', '5875.0', '5850.0']
V68 (numeric, 1048 distinct): ['10.5111', '34.2', '29.0', '29.9', '34.1', '27.7', '34.0', '29.3', '33.4', '24.25']
V69 (numeric, 658 distinct): ['37.3883', '45.0', '43.7', '44.6', '45.9', '44.9', '43.6', '41.7', '36.4', '44.8']
V70 (numeric, 72 distinct): ['10150.0', '10165.0', '10140.0', '10155.0', '10145.0', '10160.0', '10135.0', '10164.1984', '10130.0', '10170.0']
V71 (numeric, 57 distinct): ['0.0', '-5.0', '-15.0', '-10.0', '10.0', '-0.1199', '5.0', '-20.0', '15.0', '-25.0']
V72 (numeric, 175 distinct): ['0.0', '0.03', '0.05', '0.08', '0.18', '0.1', '0.13', '0.23', '0.15', '0.25']

Dataset Name: ozone_level
====
Examples: 2536
====
URL: https://www.openml.org/search?type=data&id=301
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

1. Title: Ozone Level Detection
2. Source:
Kun Zhang
zhang.kun05 '@' gmail.com
Department of Computer Science, 
Xavier University of Lousiana

Wei Fan
wei.fan '@' gmail.com
IBM T.J.Watson Research

XiaoJing Yuan
xyuan '@' uh.edu
Engineering Technology Department, 
College of Technology, University of Houston 

3. Past Usage:
Forecasting skewed biased stochastic ozone days: analyses, solutions and beyond, Knowledge and Information Systems, Vol. 14, No. 3, 2008.
Discusses details about the dataset, its use as well as various experiments (both cross-validation and streaming) using many state-of-the-art methods.

A shorter version of the paper (does not contain some detailed experiments as the journal paper above) is in:
Forecasting Skewed Biased Stochastic Ozone Days: Analyses and Solutions. ICDM 2006: 753-764 

4. Relevant Information:
The following are specifications for several most important attributes 
that are highly valued by Texas Commission on Environmental Quality (TCEQ). 
More details can be found in the two relevant papers.
 
-- O 3 - Local ozone peak prediction
-- Upwind - Upwind ozone background level
-- EmFactor - Precursor emissions related factor
-- Tmax - Maximum temperature in degrees F
-- Tb - Base temperature where net ozone production begins (50 F)
-- SRd - Solar radiation total for the day
-- WSa - Wind speed near sunrise (using 09-12 UTC forecast mode)
-- WSp - Wind speed mid-day (using 15-21 UTC forecast mode) 

5. Number of Instances: 2536

6. Number of Attributes: 73

7. Attribute Information:
1,0 | two classes 1: ozone day, 0: normal day
====
Target Variable: Class (numeric, 2 distinct): ['0', '1']
====
Features:

WSR0 (nominal, 69 distinct): ['?', '0.4', '0.8', '0.3', '1.3', '1.7', '1.1', '0.2', '0.5', '1']
WSR1 (nominal, 71 distinct): ['?', '0.4', '0.8', '0.3', '0.2', '1.3', '0.6', '0.1', '0.7', '0.9']
WSR2 (nominal, 66 distinct): ['?', '0.4', '0.8', '0.3', '1.3', '0.1', '0.2', '0.9', '0.7', '1']
WSR3 (nominal, 67 distinct): ['?', '0.4', '0.8', '0.3', '1.3', '0.2', '0.6', '0.1', '1.1', '0.5']
WSR4 (nominal, 65 distinct): ['?', '0.4', '0.8', '0.2', '0.3', '1.3', '1.7', '0.5', '0.6', '1']
WSR5 (nominal, 64 distinct): ['?', '0.4', '0.8', '1.3', '1.4', '0.7', '0.2', '1', '0.5', '0.6']
WSR6 (nominal, 67 distinct): ['?', '1.3', '0.8', '1.2', '0.4', '1.7', '1.4', '0.3', '0.5', '1']
WSR7 (nominal, 68 distinct): ['?', '1.7', '1.3', '2.1', '1.5', '1.4', '0.8', '2.5', '1.6', '1.9']
WSR8 (nominal, 70 distinct): ['?', '2.1', '3', '1.3', '2.5', '2.7', '2.4', '3.4', '1.7', '2']
WSR9 (nominal, 71 distinct): ['?', '3.4', '2.5', '3', '2.1', '3.8', '2.7', '1.7', '2.8', '3.2']
WSR10 (nominal, 77 distinct): ['?', '2.5', '3.8', '2.1', '3.4', '3', '1.7', '2.6', '2.8', '2.2']
WSR11 (nominal, 78 distinct): ['?', '2.1', '2.5', '3.8', '3', '3.4', '3.6', '2.6', '1.7', '2.3']
WSR12 (nominal, 78 distinct): ['?', '3.4', '2.1', '2.5', '3', '3.8', '2.8', '2.2', '3.3', '4.2']
WSR13 (nominal, 79 distinct): ['?', '3', '2.1', '2.5', '3.4', '3.8', '2.3', '4.2', '3.5', '1.7']
WSR14 (nominal, 78 distinct): ['?', '3', '3.4', '2.5', '3.8', '4.2', '2.1', '1.7', '2.4', '4.6']
WSR15 (nominal, 79 distinct): ['?', '3.4', '3.8', '3', '2.5', '3.7', '4.2', '2.1', '2', '4']
WSR16 (nominal, 73 distinct): ['?', '3.4', '2.5', '4.2', '3', '3.8', '2.1', '3.7', '3.6', '2.6']
WSR17 (nominal, 74 distinct): ['?', '2.1', '3.4', '3.8', '2.5', '3', '2.6', '2.4', '3.2', '1.7']
WSR18 (nominal, 71 distinct): ['?', '2.5', '2.1', '1.3', '1.7', '3.4', '3', '2.4', '2.2', '3.8']
WSR19 (nominal, 66 distinct): ['?', '1.7', '2.5', '2.1', '3', '1.3', '2', '2.7', '3.4', '1.4']
WSR20 (nominal, 69 distinct): ['?', '1.7', '1.3', '2.1', '3', '2.5', '1.4', '1.8', '1.6', '1.5']
WSR21 (nominal, 70 distinct): ['?', '1.3', '1.7', '1.6', '2.1', '1.5', '1', '1.2', '1.4', '1.9']
WSR22 (nominal, 69 distinct): ['?', '1.3', '1.7', '0.8', '1.2', '0.4', '2.1', '1', '0.9', '1.4']
WSR23 (nominal, 66 distinct): ['?', '1.3', '0.4', '0.8', '1.1', '0.9', '0.3', '1', '1.6', '0.2']
WSR_PK (nominal, 75 distinct): ['?', '3.4', '3.8', '4.2', '4.6', '4.1', '4.4', '3.6', '4.3', '3.9']
WSR_AV (nominal, 56 distinct): ['?', '1.5', '1.6', '1.9', '1.7', '2.1', '2', '1.8', '1.4', '2.4']
T0 (nominal, 283 distinct): ['?', '26.2', '25.2', '26.4', '25.1', '25.8', '26.6', '26.7', '26.1', '23.3']
T1 (nominal, 285 distinct): ['?', '26.4', '24.9', '25.6', '26.2', '26.1', '24.1', '25.9', '22.6', '25.8']
T2 (nominal, 288 distinct): ['?', '26.1', '25.9', '24.9', '21.8', '24.8', '25.3', '25.6', '25.8', '25.4']
T3 (nominal, 284 distinct): ['?', '24.7', '25.2', '25.7', '26.1', '25.4', '25.9', '25.1', '23.2', '25.6']
T4 (nominal, 284 distinct): ['?', '25.7', '26.1', '25.2', '24.3', '25.8', '24.8', '25.1', '24.2', '25.6']
T5 (nominal, 293 distinct): ['?', '24.1', '25.2', '24.7', '25.8', '24.9', '25.6', '25.1', '21.2', '23.2']
T6 (nominal, 296 distinct): ['?', '25.1', '26.1', '25.7', '25.6', '25.4', '24.9', '25.9', '25.2', '24.2']
T7 (nominal, 312 distinct): ['?', '26.6', '27.1', '26.8', '27.7', '20.1', '26.4', '25.4', '27.3', '23.2']
T8 (nominal, 314 distinct): ['?', '28.6', '28.3', '27.9', '27.8', '28.8', '28.4', '28.1', '27.6', '28.2']
T9 (nominal, 315 distinct): ['?', '29.4', '29.9', '29.8', '30.2', '29.2', '28.9', '28.1', '29.7', '29.6']
T10 (nominal, 328 distinct): ['?', '30.3', '31.1', '30.9', '30.2', '25.6', '30.6', '30.8', '30.7', '30.4']
T11 (nominal, 331 distinct): ['?', '31.1', '31.3', '31.4', '31.2', '32.3', '32.2', '24.6', '32.1', '24.7']
T12 (nominal, 335 distinct): ['?', '31.8', '33.2', '26.6', '30.8', '32.2', '30.4', '32.9', '32.7', '31.2']
T13 (nominal, 336 distinct): ['?', '30.2', '32.2', '30.1', '24.7', '24.8', '26.3', '27.4', '29.2', '30.4']
T14 (nominal, 336 distinct): ['?', '26.1', '29.6', '24.4', '26.7', '25.2', '30.4', '24.1', '30.9', '30.7']
T15 (nominal, 340 distinct): ['?', '24.1', '30.6', '24.8', '30.2', '27.8', '23.4', '32.3', '31.3', '29.3']
T16 (nominal, 338 distinct): ['?', '30.7', '26.6', '25.4', '25.2', '24.8', '24.4', '29.3', '29.1', '25.6']
T17 (nominal, 330 distinct): ['?', '28.9', '24.1', '24.3', '30.9', '30.6', '24.6', '28.7', '31.7', '22.1']
T18 (nominal, 322 distinct): ['?', '24.2', '25.3', '29.3', '27.4', '23.4', '30.7', '26.9', '26.6', '29.1']
T19 (nominal, 307 distinct): ['?', '24.4', '26.3', '25.8', '29.4', '22.4', '26.1', '28.7', '25.1', '29.2']
T20 (nominal, 303 distinct): ['?', '20.7', '28.6', '28.1', '28.7', '24.1', '28.8', '26.1', '21.8', '27.9']
T21 (nominal, 295 distinct): ['?', '26.8', '25.8', '20.1', '27.3', '25.9', '24.3', '28.1', '26.7', '23.6']
T22 (nominal, 288 distinct): ['?', '27.1', '26.1', '26.9', '22.7', '24.2', '26.6', '24.3', '25.9', '26.3']
T23 (nominal, 285 distinct): ['?', '26.6', '27.2', '25.6', '26.4', '26.7', '26.3', '27.3', '25.9', '26.2']
T_PK (nominal, 331 distinct): ['?', '30.8', '31.8', '31.7', '33.6', '26.1', '30.9', '31.1', '29.6', '32.9']
T_AV (nominal, 297 distinct): ['?', '28.7', '28.4', '27.2', '26.9', '26.3', '28.5', '27.3', '28.8', '28.9']
T85 (nominal, 252 distinct): ['?', '18.1', '16.8', '17.1', '17.5', '18.3', '17.4', '18', '16.9', '17.8']
RH85 (nominal, 101 distinct): ['?', '0.8', '0.79', '0.76', '0.86', '0.64', '0.66', '0.83', '0.77', '0.81']
U85 (nominal, 1290 distinct): ['?', '0', '-2.73', '1.93', '0.29', '1.9', '1.06', '0.26', '-0.63', '2.43']
V85 (nominal, 1463 distinct): ['?', '-1.76', '0.42', '-0.15', '2.54', '4.7', '-4.23', '2.6', '2.15', '0.27']
HT85 (nominal, 369 distinct): ['?', '1552', '1517', '1520', '1532', '1534.5', '1565.5', '1519.5', '1538.5', '1511.5']
T70 (nominal, 246 distinct): ['?', '9.1', '7.3', '9', '8.8', '8.9', '8.1', '8.4', '8.5', '8.2']
RH70 (nominal, 101 distinct): ['?', '0.02', '0.03', '0.07', '0.09', '0.2', '0.28', '0.12', '0.15', '0.53']
U70 (nominal, 1538 distinct): ['?', '0.53', '8.23', '3.26', '11.82', '0.74', '3.27', '5.35', '10.36', '11.62']
V70 (nominal, 1430 distinct): ['?', '0', '4.88', '-1.14', '-0.8', '-0.66', '1.99', '-2.28', '2.05', '0.18']
HT70 (nominal, 442 distinct): ['?', '3165.5', '3150.5', '3160.5', '3166.5', '3170', '3194', '3167', '3157', '3153.5']
T50 (nominal, 187 distinct): ['?', '-6.8', '-7.6', '-8.2', '-6.6', '-6.7', '-7.9', '-7.5', '-7', '-7.3']
RH50 (nominal, 101 distinct): ['?', '0.03', '0.04', '0.02', '0.05', '0.06', '0.09', '0.15', '0.1', '0.08']
U50 (nominal, 1688 distinct): ['?', '7.9', '7.85', '8.45', '-0.89', '9.05', '16.93', '13.28', '14.88', '4.21']
V50 (nominal, 1511 distinct): ['?', '0', '0.49', '-2.46', '-0.27', '-0.94', '-0.75', '-2.79', '4.56', '-0.92']
HT50 (nominal, 86 distinct): ['?', '5885', '5895', '5880', '5900', '5865', '5855', '5860', '5875', '5850']
KI (nominal, 1049 distinct): ['?', '34.1', '29', '29.9', '34.2', '27.7', '24.25', '29.3', '32', '30.8']
TT (nominal, 658 distinct): ['?', '45', '43.7', '44.6', '45.9', '44.9', '43.6', '44.7', '42.4', '44.5']
SLP (nominal, 72 distinct): ['10150', '10165', '10140', '10155', '10145', '10160', '10135', '?', '10130', '10170']
SLP_ (nominal, 57 distinct): ['0', '-5', '-15', '-10', '10', '?', '5', '-20', '15', '-25']
Precp (nominal, 175 distinct): ['0', '0.03', '0.05', '0.08', '0.18', '0.1', '0.13', '0.15', '0.23', '0.25']
'''

CONTEXT = "Ozone Level Detection"
TARGET = CuratedTarget(raw_name="Class", new_name="Ozone Level", task_type=SupervisedTask.BINARY,
                       label_mapping={'1': 'Normal Day', '2': 'Ozone Day'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Wind Speed at 0"),
            CuratedFeature(raw_name="V2", new_name="Wind Speed at 1"),
            CuratedFeature(raw_name="V3", new_name="Wind Speed at 2"),
            CuratedFeature(raw_name="V4", new_name="Wind Speed at 3"),
            CuratedFeature(raw_name="V5", new_name="Wind Speed at 4"),
            CuratedFeature(raw_name="V6", new_name="Wind Speed at 5"),
            CuratedFeature(raw_name="V7", new_name="Wind Speed at 6"),
            CuratedFeature(raw_name="V8", new_name="Wind Speed at 7"),
            CuratedFeature(raw_name="V9", new_name="Wind Speed at 8"),
            CuratedFeature(raw_name="V10", new_name="Wind Speed at 9"),
            CuratedFeature(raw_name="V11", new_name="Wind Speed at 10"),
            CuratedFeature(raw_name="V12", new_name="Wind Speed at 11"),
            CuratedFeature(raw_name="V13", new_name="Wind Speed at 12"),
            CuratedFeature(raw_name="V14", new_name="Wind Speed at 13"),
            CuratedFeature(raw_name="V15", new_name="Wind Speed at 14"),
            CuratedFeature(raw_name="V16", new_name="Wind Speed at 15"),
            CuratedFeature(raw_name="V17", new_name="Wind Speed at 16"),
            CuratedFeature(raw_name="V18", new_name="Wind Speed at 17"),
            CuratedFeature(raw_name="V19", new_name="Wind Speed at 18"),
            CuratedFeature(raw_name="V20", new_name="Wind Speed at 19"),
            CuratedFeature(raw_name="V21", new_name="Wind Speed at 20"),
            CuratedFeature(raw_name="V22", new_name="Wind Speed at 21"),
            CuratedFeature(raw_name="V23", new_name="Wind Speed at 22"),
            CuratedFeature(raw_name="V24", new_name="Wind Speed at 23"),
            CuratedFeature(raw_name="V25", new_name="Wind Speed Peak"),
            CuratedFeature(raw_name="V26", new_name="Wind Speed Average"),
            CuratedFeature(raw_name="V27", new_name="Temperature at 0"),
            CuratedFeature(raw_name="V28", new_name="Temperature at 1"),
            CuratedFeature(raw_name="V29", new_name="Temperature at 2"),
            CuratedFeature(raw_name="V30", new_name="Temperature at 3"),
            CuratedFeature(raw_name="V31", new_name="Temperature at 4"),
            CuratedFeature(raw_name="V32", new_name="Temperature at 5"),
            CuratedFeature(raw_name="V33", new_name="Temperature at 6"),
            CuratedFeature(raw_name="V34", new_name="Temperature at 7"),
            CuratedFeature(raw_name="V35", new_name="Temperature at 8"),
            CuratedFeature(raw_name="V36", new_name="Temperature at 9"),
            CuratedFeature(raw_name="V37", new_name="Temperature at 10"),
            CuratedFeature(raw_name="V38", new_name="Temperature at 11"),
            CuratedFeature(raw_name="V39", new_name="Temperature at 12"),
            CuratedFeature(raw_name="V40", new_name="Temperature at 13"),
            CuratedFeature(raw_name="V41", new_name="Temperature at 14"),
            CuratedFeature(raw_name="V42", new_name="Temperature at 15"),
            CuratedFeature(raw_name="V43", new_name="Temperature at 16"),
            CuratedFeature(raw_name="V44", new_name="Temperature at 17"),
            CuratedFeature(raw_name="V45", new_name="Temperature at 18"),
            CuratedFeature(raw_name="V46", new_name="Temperature at 19"),
            CuratedFeature(raw_name="V47", new_name="Temperature at 20"),
            CuratedFeature(raw_name="V48", new_name="Temperature at 21"),
            CuratedFeature(raw_name="V49", new_name="Temperature at 22"),
            CuratedFeature(raw_name="V50", new_name="Temperature at 23"),
            CuratedFeature(raw_name="V51", new_name="Temperature Peak"),
            CuratedFeature(raw_name="V52", new_name="Temperature Average"),
            CuratedFeature(raw_name="V53", new_name="Temperature 85"),
            CuratedFeature(raw_name="V54", new_name="Relative Humidity 85"),
            CuratedFeature(raw_name="V55", new_name="U 85"),
            CuratedFeature(raw_name="V56", new_name="V 85"),
            CuratedFeature(raw_name="V57", new_name="HT 85"),
            CuratedFeature(raw_name="V58", new_name="Temperature 70"),
            CuratedFeature(raw_name="V59", new_name="Relative Humidity 70"),
            CuratedFeature(raw_name="V60", new_name="U 70"),
            CuratedFeature(raw_name="V61", new_name="V 70"),
            CuratedFeature(raw_name="V62", new_name="HT 70"),
            CuratedFeature(raw_name="V63", new_name="Temperature 50"),
            CuratedFeature(raw_name="V64", new_name="Relative Humidity 50"),
            CuratedFeature(raw_name="V65", new_name="U 50"),
            CuratedFeature(raw_name="V66", new_name="V 50"),
            CuratedFeature(raw_name="V67", new_name="HT 50"),
            CuratedFeature(raw_name="V68", new_name="K-Index"),
            CuratedFeature(raw_name="V69", new_name="T-Totals"),
            CuratedFeature(raw_name="V70", new_name="Sea Level Pressure"),
            CuratedFeature(raw_name="V71", new_name="Sea Level Pressure change from previous day"),
            CuratedFeature(raw_name="V72", new_name="Precipitation")
            ]
