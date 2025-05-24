from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: led24
====
Examples: 3200
====
URL: https://www.openml.org/search?type=data&id=40677
====
Description: led24-pmlb
===
Other:
https://www.openml.org/search?type=data&status=active&id=40496&sort=runs

Description
Author: Breiman,L., Friedman,J.H., Olshen,R.A., and Stone,C.J.
Source: UCI, [KEEL](http://sci2s.ugr.es/keel/dataset.php?cod=63, https://archive.ics.uci.edu/ml/datasets/LED+Display+Domain) - 1988
Please cite: UCI

LED display data set
This simple domain contains 7 Boolean attributes and 10 classes, the set of decimal digits. Recall that LED displays contain 7 light-emitting diodes -- hence the reason for 7 attributes. The class attribute is an integer ranging between 0 and 9 inclusive, representing the possible digits show on the display.

The problem would be easy if not for the introduction of noise. In this case, each attribute value has the 10% probability of having its value inverted.

It's valuable to know the optimal Bayes rate for these databases. In this case, the misclassification rate is 26% (74% classification accuracy).

Attribute Information
V1-V7 represent each of the 7 LEDs, with values either 0 or 1, according to whether the corresponding light is on or not for the decimal digit. Each has a 10% percent chance of being inverted.

====
Target Variable: class (nominal, 10 distinct): ['6', '3', '2', '0', '5', '7', '9', '8', '1', '4']
====
Features:

attribute#1 (nominal, 2 distinct): ['1', '0']
attribute#2 (nominal, 2 distinct): ['1', '0']
attribute#3 (nominal, 2 distinct): ['1', '0']
attribute#4 (nominal, 2 distinct): ['1', '0']
attribute#5 (nominal, 2 distinct): ['0', '1']
attribute#6 (nominal, 2 distinct): ['1', '0']
attribute#7 (nominal, 2 distinct): ['1', '0']
irrelevant1 (nominal, 2 distinct): ['0', '1']
irrelevant2 (nominal, 2 distinct): ['1', '0']
irrelevant3 (nominal, 2 distinct): ['1', '0']
irrelevant4 (nominal, 2 distinct): ['0', '1']
irrelevant5 (nominal, 2 distinct): ['1', '0']
irrelevant6 (nominal, 2 distinct): ['0', '1']
irrelevant7 (nominal, 2 distinct): ['0', '1']
irrelevant8 (nominal, 2 distinct): ['0', '1']
irrelevant9 (nominal, 2 distinct): ['0', '1']
irrelevant10 (nominal, 2 distinct): ['0', '1']
irrelevant11 (nominal, 2 distinct): ['0', '1']
irrelevant12 (nominal, 2 distinct): ['1', '0']
irrelevant13 (nominal, 2 distinct): ['0', '1']
irrelevant14 (nominal, 2 distinct): ['0', '1']
irrelevant15 (nominal, 2 distinct): ['0', '1']
irrelevant16 (nominal, 2 distinct): ['0', '1']
irrelevant17 (nominal, 2 distinct): ['1', '0']
'''

CONTEXT = "Anonymized: LED24"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []