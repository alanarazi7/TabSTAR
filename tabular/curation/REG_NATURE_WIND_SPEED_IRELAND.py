from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wind
====
Examples: 6574
====
URL: https://www.openml.org/search?type=data&id=503
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

wind   daily average wind speeds for 1961-1978 at 12 synoptic meteorological
stations in the Republic of Ireland (Haslett and raftery 1989).

These data were analyzed in detail in the following article:
Haslett, J. and Raftery, A. E. (1989). Space-time Modelling with
Long-memory Dependence: Assessing Ireland's Wind Power Resource
(with Discussion). Applied Statistics 38, 1-50.

Each line corresponds to one day of data in the following format:
year, month, day, average wind speed at each of the stations in the order given
in Fig.4 of Haslett and Raftery :
RPT, VAL, ROS, KIL, SHA, BIR, DUB, CLA, MUL, CLO, BEL, MAL

Fortan format : ( i2, 2i3, 12f6.2)

The data are in knots, not in m/s.

Permission granted for unlimited distribution.

Please report all anomalies to fraley@stat.washington.edu

Be aware that the dataset is 532494 bytes long (thats over half a
Megabyte).  Please be sure you want the data before you request it.


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific
====
Target Variable: MAL (numeric, 779 distinct): ['14.67', '11.96', '13.21', '12.92', '16.08', '13.79', '14.42', '15.75', '17.75', '18.63']
====
Features:

year (numeric, 18 distinct): ['64', '76', '68', '72', '61', '71', '77', '75', '74', '73']
month (numeric, 12 distinct): ['1', '3', '5', '7', '8', '10', '12', '4', '6', '9']
day (numeric, 31 distinct): ['1', '2', '28', '27', '26', '25', '24', '23', '22', '21']
RPT (numeric, 671 distinct): ['9.79', '12.83', '12.67', '7.5', '9.0', '10.0', '16.88', '10.37', '13.62', '13.08']
VAL (numeric, 607 distinct): ['9.83', '11.54', '6.63', '6.92', '10.13', '10.08', '13.67', '7.83', '10.88', '7.5']
ROS (numeric, 611 distinct): ['8.12', '8.67', '8.75', '8.46', '9.54', '9.25', '10.54', '9.17', '9.59', '10.63']
KIL (numeric, 450 distinct): ['3.58', '3.96', '5.04', '3.21', '4.79', '4.21', '5.83', '3.88', '6.04', '4.75']
SHA (numeric, 596 distinct): ['9.46', '9.92', '10.0', '10.54', '8.96', '12.62', '8.38', '6.96', '9.96', '10.96']
BIR (numeric, 461 distinct): ['9.04', '8.29', '7.17', '4.29', '5.29', '8.67', '4.04', '5.79', '3.71', '6.25']
DUB (numeric, 580 distinct): ['8.38', '8.5', '9.42', '7.87', '10.88', '10.0', '8.29', '11.08', '8.58', '8.71']
CLA (numeric, 534 distinct): ['7.0', '7.58', '4.5', '7.75', '5.75', '5.66', '9.83', '6.58', '10.63', '6.17']
MUL (numeric, 503 distinct): ['9.0', '7.21', '9.38', '7.46', '7.12', '7.83', '6.79', '6.87', '4.46', '8.83']
CLO (numeric, 533 distinct): ['8.58', '10.83', '8.12', '6.87', '4.38', '7.29', '7.0', '8.54', '7.58', '5.75']
BEL (numeric, 687 distinct): ['10.21', '11.42', '16.38', '12.87', '10.25', '9.13', '12.5', '8.63', '12.33', '7.67']
'''

CONTEXT = "Wind speed in Ireland between 1961-1978"
TARGET = CuratedTarget(raw_name='MAL', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []