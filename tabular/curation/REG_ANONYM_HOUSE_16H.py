from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: house_16H
====
Examples: 22784
====
URL: https://www.openml.org/search?type=data&id=574
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

This database was designed on the basis of data provided by US Census
Bureau [http://www.census.gov] (under Lookup Access
[http://www.census.gov/cdrom/lookup]: Summary Tape File 1). The data
were collected as part of the 1990 US census. These are mostly counts
cumulated at different survey levels. For the purpose of this data set
a level State-Place was used. Data from all states was obtained. Most
of the counts were changed into appropriate proportions.  There are 4
different data sets obtained from this database: House(8H) House(8L)
House(16H) House(16L) These are all concerned with predicting the
median price of the house in the region based on demographic
composition and a state of housing market in the region. A number in
the name signifies the number of attributes of the data set. A
following letter denotes a very rough approximation to the difficulty
of the task. For Low task difficulty, more correlated attributes were
chosen as signified by univariate smooth fit of that input on the
target. Tasks with High difficulty have had their attributes chosen to
make the modelling more difficult due to higher variance or lower
correlation of the inputs to the target.

Original source: DELVE repository of data.
Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at
http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html
Characteristics: 22784 cases, 17 continuous attributes.
====
Target Variable: price (numeric, 2045 distinct): ['14999.0', '21300.0', '17500.0', '16300.0', '31300.0', '26300.0', '23800.0', '18800.0', '22500.0', '20000.0']
====
Features:

P1 (numeric, 8832 distinct): ['257.0', '148.0', '170.0', '192.0', '160.0', '132.0', '171.0', '88.0', '161.0', '211.0']
P5p1 (numeric, 17504 distinct): ['0.5', '0.4615', '0.4545', '0.4737', '0.4667', '0.4706', '0.4762', '0.4444', '0.4783', '0.48']
P6p2 (numeric, 13683 distinct): ['0.0', '0.0059', '0.0026', '0.0073', '0.0149', '0.0025', '0.0056', '0.0067', '0.0025', '0.0022']
P11p4 (numeric, 19220 distinct): ['0.0', '0.1667', '0.2', '0.1429', '0.1875', '0.25', '0.125', '0.1818', '0.1538', '0.1111']
P14p9 (numeric, 16168 distinct): ['0.0', '0.1667', '0.1429', '0.1111', '0.125', '0.1', '0.0833', '0.2', '0.0909', '0.1818']
P15p1 (numeric, 18753 distinct): ['0.8571', '0.875', '0.9091', '0.8889', '0.8333', '0.9', '1.0', '0.8824', '0.8947', '0.8']
P15p3 (numeric, 9655 distinct): ['0.0', '0.0455', '0.0101', '0.0156', '0.08', '0.0085', '0.0556', '0.0267', '0.0263', '0.0323']
P16p2 (numeric, 15570 distinct): ['0.75', '0.6667', '0.8', '0.7143', '0.7', '0.7778', '0.7273', '0.6', '0.8333', '1.0']
P18p2 (numeric, 8070 distinct): ['0.0', '0.0085', '0.006', '0.0062', '0.0045', '0.0057', '0.0057', '0.0059', '0.0079', '0.0132']
P27p4 (numeric, 12052 distinct): ['0.0', '0.0303', '0.0286', '0.0385', '0.0227', '0.0357', '0.0204', '0.0323', '0.0263', '0.0417']
H2p2 (numeric, 15662 distinct): ['0.0', '0.125', '0.1', '0.1111', '0.1429', '0.1667', '0.0769', '0.2', '0.0909', '0.25']
H8p2 (numeric, 10941 distinct): ['0.0', '0.5', '0.0103', '0.0052', '0.0041', '0.0286', '0.25', '0.0143', '0.02', '0.0059']
H10p1 (numeric, 10855 distinct): ['1.0', '0.9821', '0.9818', '0.9933', '0.9897', '0.9853', '0.9912', '0.9922', '0.9885', '0.9918']
H13p1 (numeric, 17097 distinct): ['0.3333', '0.25', '0.0', '0.2', '0.2857', '0.4', '0.1667', '0.5', '0.2727', '0.2222']
H18pA (numeric, 9063 distinct): ['0.0', '0.1667', '0.1429', '0.1111', '0.125', '0.2', '0.25', '0.1', '0.0909', '0.0833']
H40p4 (numeric, 2421 distinct): ['0.0', '1.0', '0.5', '0.6667', '0.3333', '0.75', '0.6', '0.8', '0.25', '0.4']
'''

CONTEXT = "US 1990 Census House Prices"
TARGET = CuratedTarget(raw_name='price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []