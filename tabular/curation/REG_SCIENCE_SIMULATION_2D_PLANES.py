from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: 2dplanes
====
Examples: 40768
====
URL: https://www.openml.org/search?type=data&id=215
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

This is an artificial data set described in Breiman et al. (1984,p.238) 
 (with variance 1 instead of 2).  
 
 Generate the values of the 10 attributes independently
 using the following probabilities:

 P(X_1 = -1) = P(X_1 = 1) = 1/2
 P(X_m = -1) = P(X_m = 0) = P(X_m = 1) = 1/3, m=2,...,10

 Obtain the value of the target variable Y using the rule:

 if X_1 = 1 set Y = 3 + 3X_2 + 2X_3 + X_4 + sigma(0,1)
 if X_1 = -1 set Y = -3 + 3X_5 + 2X_6 + X_7 + sigma(0,1)

 Characteristics: 40768 cases, 11 continuous attributes
 Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at
 http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html
 Original source: Breiman et al. (1984, p.238).
====
Target Variable: y (numeric, 40368 distinct): ['1.035', '1.8173', '2.0453', '1.5654', '-1.9915', '4.1449', '5.3322', '-1.6384', '4.918', '-2.7656']
====
Features:

x1 (numeric, 2 distinct): ['1.0', '-1.0']
x2 (numeric, 3 distinct): ['0.0', '-1.0', '1.0']
x3 (numeric, 3 distinct): ['1.0', '-1.0', '0.0']
x4 (numeric, 3 distinct): ['1.0', '0.0', '-1.0']
x5 (numeric, 3 distinct): ['1.0', '-1.0', '0.0']
x6 (numeric, 3 distinct): ['1.0', '-1.0', '0.0']
x7 (numeric, 3 distinct): ['0.0', '1.0', '-1.0']
x8 (numeric, 3 distinct): ['-1.0', '0.0', '1.0']
x9 (numeric, 3 distinct): ['1.0', '0.0', '-1.0']
x10 (numeric, 3 distinct): ['1.0', '0.0', '-1.0']
'''

CONTEXT = "2d Planes Artificial Data"
TARGET = CuratedTarget(raw_name="y", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []