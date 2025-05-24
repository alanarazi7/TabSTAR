from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: arsenic-male-lung
====
Examples: 559
====
URL: https://www.openml.org/search?type=data&id=951
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Binarized version of the original data set (see version 1). It converts the numeric target feature to a two-class nominal target feature by computing the mean and classifying all instances with a lower target value as positive ('P') and all others as negative ('N'). Originally converted by Quan Sun.
====
Target Variable: binaryClass (nominal, 2 distinct): ['P', 'N']
====
Features:

group (nominal, 43 distinct): ['1', '33', '25', '26', '27', '28', '29', '30', '31', '32']
conc (numeric, 38 distinct): ['307.0', '256.0', '32.0', '110.0', '520.0', '538.0', '448.0', '467.0', '504.0', '529.0']
age (numeric, 13 distinct): ['22.5', '27.5', '32.5', '37.5', '42.5', '47.5', '52.5', '57.5', '62.5', '67.5']
at.risk (numeric, 425 distinct): ['247.0', '122.0', '104.0', '249.0', '39.0', '22.0', '295.0', '637.0', '782.0', '80.0']
'''

CONTEXT = "Arsenic Male Lung"
TARGET = CuratedTarget(raw_name="binaryClass", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []