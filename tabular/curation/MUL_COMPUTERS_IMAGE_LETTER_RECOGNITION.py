from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: letter
====
Examples: 20000
====
URL: https://www.openml.org/search?type=data&id=6
====
Description: **Author**: David J. Slate  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) - 01-01-1991  
**Please cite**: P. W. Frey and D. J. Slate. "Letter Recognition Using Holland-style Adaptive Classifiers". Machine Learning 6(2), 1991  

1. TITLE: 
  Letter Image Recognition Data 
 
    The objective is to identify each of a large number of black-and-white
    rectangular pixel displays as one of the 26 capital letters in the English
    alphabet.  The character images were based on 20 different fonts and each
    letter within these 20 fonts was randomly distorted to produce a file of
    20,000 unique stimuli.  Each stimulus was converted into 16 primitive
    numerical attributes (statistical moments and edge counts) which were then
    scaled to fit into a range of integer values from 0 through 15.  We
    typically train on the first 16000 items and then use the resulting model
    to predict the letter category for the remaining 4000.  See the article
    cited above for more details.
====
Target Variable: class (nominal, 26 distinct): ['U', 'D', 'P', 'T', 'M', 'A', 'X', 'Y', 'Q', 'N']
====
Features:

x-box (numeric, 16 distinct): ['4', '3', '5', '2', '6', '1', '7', '8', '9', '0']
y-box (numeric, 16 distinct): ['9', '7', '10', '8', '6', '11', '5', '4', '3', '1']
width (numeric, 16 distinct): ['5', '4', '6', '3', '7', '8', '2', '9', '1', '10']
high (numeric, 16 distinct): ['6', '8', '4', '7', '5', '3', '2', '1', '0', '9']
onpix (numeric, 16 distinct): ['2', '3', '4', '1', '5', '6', '7', '0', '8', '9']
x-bar (numeric, 16 distinct): ['7', '8', '6', '5', '9', '4', '10', '3', '11', '12']
y-bar (numeric, 16 distinct): ['7', '8', '6', '9', '10', '11', '5', '4', '12', '3']
x2bar (numeric, 16 distinct): ['3', '4', '5', '2', '6', '7', '1', '8', '9', '0']
y2bar (numeric, 16 distinct): ['5', '6', '4', '7', '3', '2', '8', '9', '1', '10']
xybar (numeric, 16 distinct): ['7', '6', '10', '9', '11', '8', '12', '5', '13', '14']
x2ybr (numeric, 16 distinct): ['6', '7', '5', '8', '4', '9', '11', '10', '2', '3']
xy2br (numeric, 16 distinct): ['8', '9', '7', '6', '10', '5', '11', '4', '12', '13']
x-ege (numeric, 16 distinct): ['3', '2', '1', '0', '4', '5', '6', '7', '8', '9']
xegvy (numeric, 16 distinct): ['8', '9', '7', '10', '6', '11', '5', '12', '4', '13']
y-ege (numeric, 16 distinct): ['4', '3', '2', '0', '5', '1', '6', '7', '8', '9']
yegvx (numeric, 16 distinct): ['8', '7', '9', '6', '10', '5', '11', '4', '12', '3']
'''

CONTEXT = "Letter Image Recognition Data"
TARGET = CuratedTarget(raw_name="class", new_name="Letter", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []