from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mv
====
Examples: 40768
====
URL: https://www.openml.org/search?type=data&id=344
====
Description: **Author**: Luis Torgo  
**Source**: [original](http://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html) -   
**Please cite**:   

This is an artificial data set with dependencies between the attribute values. The cases are generated using the following method:

X1 : uniformly distributed over [-5,5]
X2 : uniformly distributed over [-15,-10]
X3 : IF (X1 > 0) THEN X3 = green
 ELSE X3 = red with probability 0.4 and X4=brown with prob. 0.6
X4 : IF (X3=green) THEN X4=X1+2X2
 ELSE X4=X1/2 with prob. 0.3, and X4=X2/2 with prob. 0.7
X5 : uniformly distributed over [-1,1]
X6 : X6=X4*[epsilon], where [epsilon] is uniformly distribute over [0,5]
X7 : X7=yes with prob. 0.3 and X7=no with prob. 0.7
X8 : IF (X5 < 0.5) THEN X8 = normal ELSE X8 = large
X9 : uniformly distributed over [100,500]
X10 : uniformly distributed integer over the interval [1000,1200]
 
Obtain the value of the target variable Y using the rules:
IF (X2 > 2 ) THEN Y = 35 - 0.5 X4
 ELSE IF (-2 <= X4 <= 2) THEN Y = 10 - 2 X1
 ELSE IF (X7 = yes) THEN Y = 3 -X1/X4
 ELSE IF (X8 = normal) THEN Y = X6 + X1
 ELSE Y = X1/2

Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at http://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html
====
Target Variable: y (numeric, 39029 distinct): ['1.0', '-16.1812', '-15.4824', '-20.8324', '-14.0139', '-13.7257', '-14.5302', '-12.6612', '-21.1923', '2.0749']
====
Features:

x1 (numeric, 40105 distinct): ['4.1498', '3.4488', '4.3179', '-1.7334', '-1.77', '3.0906', '-3.4205', '-2.001', '3.5363', '1.5943']
x2 (numeric, 27796 distinct): ['-14.4027', '-11.6732', '-13.7954', '-13.1298', '-11.4907', '-14.537', '-10.8731', '-14.0444', '-14.5904', '-12.0383']
x3 (nominal, 3 distinct): ['brown', 'red', 'green']
x4 (numeric, 39011 distinct): ['-6.2171', '-7.4843', '-6.3386', '-5.1424', '-7.3358', '-5.5007', '-6.2045', '-6.672', '-7.4962', '-6.1618']
x5 (numeric, 40418 distinct): ['0.1716', '-0.103', '-0.6011', '-0.1011', '-0.6111', '-0.5907', '-0.4346', '-0.5527', '-0.7341', '0.2479']
x6 (numeric, 39833 distinct): ['-22.2455', '-22.5169', '-14.7099', '-25.2976', '-14.0732', '-11.3561', '-18.2734', '-13.7097', '-15.0041', '-12.8678']
x7 (nominal, 2 distinct): ['no', 'yes']
x8 (nominal, 2 distinct): ['normal', 'large']
x9 (numeric, 38738 distinct): ['120.675', '124.223', '155.048', '186.659', '165.266', '364.851', '266.313', '120.936', '140.235', '348.468']
x10 (numeric, 201 distinct): ['1039.0', '1149.0', '1191.0', '1186.0', '1102.0', '1139.0', '1021.0', '1182.0', '1178.0', '1046.0']
'''

CONTEXT = "Synthetic Distribution MV"
TARGET = CuratedTarget(raw_name="y", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []