from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: airlines
====
Examples: 539383
====
URL: https://www.openml.org/search?type=data&id=1169
====
Description: **Author**: Albert Bifet, Elena Ikonomovska  
**Source**: [Data Expo competition](http://kt.ijs.si/elena_ikonomovska/data.html) - 2009  
**Please cite**:   

Airlines Dataset Inspired in the regression dataset from Elena Ikonomovska. The task is to predict whether a given flight will be delayed, given the information of the scheduled departure.
====
Target Variable: Delay (nominal, 2 distinct): ['0', '1']
====
Features:

Airline (nominal, 18 distinct): ['WN', 'DL', 'OO', 'AA', 'MQ', 'US', 'XE', 'EV', 'UA', 'CO']
Flight (numeric, 6585 distinct): ['16.0', '5.0', '9.0', '8.0', '62.0', '371.0', '28.0', '12.0', '511.0', '55.0']
AirportFrom (nominal, 293 distinct): ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'IAH', 'PHX', 'DTW', 'LAS', 'SFO']
AirportTo (nominal, 293 distinct): ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'IAH', 'PHX', 'DTW', 'LAS', 'SFO']
DayOfWeek (nominal, 7 distinct): ['4', '3', '5', '1', '2', '7', '6']
Time (numeric, 1131 distinct): ['360.0', '420.0', '390.0', '480.0', '450.0', '540.0', '510.0', '1020.0', '660.0', '990.0']
Length (numeric, 426 distinct): ['80.0', '70.0', '65.0', '85.0', '75.0', '90.0', '60.0', '95.0', '125.0', '110.0']
'''

CONTEXT = "Airlines Departure Delay"
TARGET = CuratedTarget(raw_name="Delay", task_type=SupervisedTask.BINARY, label_mapping={'0': 'On Time', '1': 'Delayed'})
COLS_TO_DROP = []
FEATURES = []
