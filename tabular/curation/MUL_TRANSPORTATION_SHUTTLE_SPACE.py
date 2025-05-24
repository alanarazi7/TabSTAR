from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: shuttle
====
Examples: 58000
====
URL: https://www.openml.org/search?type=data&id=40685
====
Description: Source: [UCI](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle))

Donor:

Jason Catlett
Basser Department of Computer Science,
University of Sydney, N.S.W., Australia



Data Set Information:

Approximately 80% of the data belongs to class 1. Therefore the default accuracy is about 80%. The aim here is to obtain an accuracy of 99 - 99.9%.

The examples in the original dataset were in time order, and this time order could presumably be relevant in classification. However, this was not deemed relevant for StatLog purposes, so the order of the examples in the original dataset was randomised, and a portion of the original dataset removed for validation purposes.


Attribute Information:

The shuttle dataset contains 9 attributes all of which are numerical. The first one being time. The last column is the class which has been coded as follows :
1 Rad Flow
2 Fpv Close
3 Fpv Open
4 High
5 Bypass
6 Bpv Close
7 Bpv Open


Relevant Papers:

N/A
====
Target Variable: class (nominal, 7 distinct): ['1', '4', '5', '3', '2', '7', '6']
====
Features:

A1 (numeric, 76 distinct): ['37', '55', '56', '41', '45', '44', '49', '43', '46', '51']
A2 (numeric, 206 distinct): ['0.0', '-1.0', '1.0', '2.0', '-2.0', '3.0', '4.0', '5.0', '-3.0', '-4.0']
A3 (numeric, 51 distinct): ['77', '81', '79', '86', '76', '83', '78', '84', '80', '88']
A4 (numeric, 137 distinct): ['0.0', '-1.0', '1.0', '-2.0', '2.0', '-3.0', '3.0', '-4.0', '4.0', '5.0']
A5 (numeric, 54 distinct): ['46.0', '42.0', '44.0', '38.0', '50.0', '54.0', '52.0', '36.0', '56.0', '34.0']
A6 (numeric, 299 distinct): ['0.0', '-1.0', '1.0', '-4.0', '-3.0', '-2.0', '-6.0', '6.0', '3.0', '5.0']
A7 (numeric, 86 distinct): ['40.0', '41.0', '42.0', '39.0', '38.0', '43.0', '37.0', '35.0', '36.0', '33.0']
A8 (numeric, 123 distinct): ['37.0', '39.0', '41.0', '35.0', '44.0', '42.0', '32.0', '34.0', '46.0', '43.0']
A9 (numeric, 77 distinct): ['0.0', '2.0', '6.0', '8.0', '4.0', '14.0', '12.0', '16.0', '10.0', '32.0']
'''

CONTEXT = "Shuttle Space"
TARGET = CuratedTarget(raw_name="class", new_name="Shuttle Class", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'Rad Flow', '2': 'Fpv Close', '3': 'Fpv Open', '4': 'High',
                                      '5': 'Bypass', '6': 'Bpv Close', '7': 'Bpv Open'})
COLS_TO_DROP = []
FEATURES = []