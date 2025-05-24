from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: rmftsa_sleepdata
====
Examples: 1024
====
URL: https://www.openml.org/search?type=data&id=679
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

File: ../data/rmftsa/sleepdata.txt

Sleep state measurements of a newborn infant (column 2) together
with his heart rate (column 1) and temperature (column 3).


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific
====
Target Variable: sleep_state (nominal, 4 distinct): ['1', '4', '3', '2']
====
Features:

heart_rate (numeric, 77 distinct): ['124', '128', '129', '130', '125', '126', '120', '123', '131', '127']
temperature (numeric, 13 distinct): ['37.05', '37.15', '37.3', '36.95', '37.2', '37.35', '37.1', '36.9', '37.25', '37.0']
'''

CONTEXT = "Newborn Baby Sleep States"
TARGET = CuratedTarget(raw_name="sleep_state", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'Active Sleep', '2': 'Quiet Sleep', '3': 'Transition', '4': 'Indeterminate'})
COLS_TO_DROP = []
FEATURES = []
