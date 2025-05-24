from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: pol
====
Examples: 15000
====
URL: https://www.openml.org/search?type=data&id=201
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

This is a commercial application described in Weiss & Indurkhya (1995). 
 The data describes a telecommunication problem. No further information
 is available.
 
 Characteristics: (10000+5000) cases, 49 continuous attributes 
 Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at
 http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html
 Original Source: The data in the original format can be obtained 
 from http://www.cs.su.oz.au/~nitin
====
Target Variable: foo (numeric, 11 distinct): ['0', '100', '90', '10', '20', '30', '50', '70', '80', '60']
====
Features:

f1 (numeric, 1 distinct): ['110']
f2 (numeric, 1 distinct): ['100']
f3 (numeric, 1 distinct): ['100']
f4 (numeric, 1 distinct): ['100']
f5 (numeric, 184 distinct): ['74', '59', '56', '76', '71', '61', '18', '77', '117', '16']
f6 (numeric, 118 distinct): ['77', '78', '79', '81', '80', '82', '83', '98', '85', '96']
f7 (numeric, 114 distinct): ['76', '77', '78', '79', '80', '81', '82', '83', '84', '85']
f8 (numeric, 106 distinct): ['71', '72', '73', '74', '75', '76', '77', '78', '79', '81']
f9 (numeric, 80 distinct): ['94', '95', '97', '96', '98', '99', '100', '101', '102', '103']
f10 (numeric, 1 distinct): ['0']
f11 (numeric, 1 distinct): ['0']
f12 (numeric, 1 distinct): ['0']
f13 (numeric, 97 distinct): ['0', '1', '2', '3', '5', '37', '6', '19', '32', '9']
f14 (numeric, 117 distinct): ['0', '1', '2', '3', '5', '37', '32', '6', '19', '42']
f15 (numeric, 121 distinct): ['0', '1', '2', '3', '5', '37', '32', '6', '4', '14']
f16 (numeric, 120 distinct): ['0', '1', '2', '3', '5', '37', '6', '32', '14', '4']
f17 (numeric, 120 distinct): ['0', '1', '2', '3', '5', '6', '37', '4', '31', '9']
f18 (numeric, 123 distinct): ['0', '1', '2', '3', '5', '6', '4', '36', '7', '44']
f19 (numeric, 102 distinct): ['0', '1', '2', '4', '3', '6', '5', '9', '25', '19']
f20 (numeric, 86 distinct): ['0', '1', '2', '3', '4', '6', '5', '15', '31', '26']
f21 (numeric, 85 distinct): ['0', '1', '2', '4', '3', '6', '5', '11', '22', '19']
f22 (numeric, 88 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '37', '16']
f23 (numeric, 79 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
f24 (numeric, 63 distinct): ['0', '1', '2', '4', '3', '5', '6', '7', '8', '20']
f25 (numeric, 68 distinct): ['0', '1', '2', '3', '4', '5', '8', '14', '6', '17']
f26 (numeric, 68 distinct): ['0', '1', '2', '4', '3', '6', '5', '7', '11', '8']
f27 (numeric, 65 distinct): ['0', '1', '2', '3', '4', '5', '7', '8', '6', '14']
f28 (numeric, 64 distinct): ['0', '1', '2', '4', '3', '5', '6', '7', '8', '10']
f29 (numeric, 62 distinct): ['0', '1', '2', '3', '4', '6', '5', '7', '11', '8']
f30 (numeric, 44 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
f31 (numeric, 43 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '11', '14']
f32 (numeric, 42 distinct): ['0', '1', '2', '3', '5', '4', '6', '7', '23', '12']
f33 (numeric, 38 distinct): ['0', '1', '2', '4', '3', '6', '5', '13', '10', '8']
f34 (numeric, 1 distinct): ['0']
f35 (numeric, 1 distinct): ['0']
f36 (numeric, 1 distinct): ['0']
f37 (numeric, 1 distinct): ['0']
f38 (numeric, 1 distinct): ['0']
f39 (numeric, 1 distinct): ['0']
f40 (numeric, 1 distinct): ['0']
f41 (numeric, 1 distinct): ['0']
f42 (numeric, 1 distinct): ['0']
f43 (numeric, 1 distinct): ['0']
f44 (numeric, 1 distinct): ['0']
f45 (numeric, 1 distinct): ['0']
f46 (numeric, 1 distinct): ['0']
f47 (numeric, 1 distinct): ['0']
f48 (numeric, 1 distinct): ['0']
'''

CONTEXT = "Telecommunication Problem"
TARGET = CuratedTarget(raw_name="foo", task_type=SupervisedTask.MULTICLASS)

COLS_TO_DROP = []
FEATURES = []