from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: solar_flare
====
Examples: 1066
====
URL: https://www.openml.org/search?type=data&id=44966
====
Description: **Data Description**

Predict the number of common solar flares.

The database contains 3 classes, one for the number of times a certain type of solar flare occurred in a 24 hour period.

Each instance represents captured features for 1 active region on the sun.

**Attribute Description**

1. *class* - code for class (modified Zurich class) (A,B,C,D,E,F,H)
2. *largest_spot_size* -  Code for largest spot size (X,R,S,A,H,K)
3. *spot_distribution* - Code for spot distribution (X,O,I,C)
4. *activity* - activity (1 = reduced, 2 = unchanged)
5. *evolution* - evolution (1 = decay, 2 = no growth, 3 = growth)
6. *previous_activity* - previous 24 hour flare activity code (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
7. *complex* - historically-complex (1 = Yes, 2 = No)
8. *complex_path* - whether region become historically complex on this pass across the sun's disk (1 = yes, 2 = no)
9. *area* - area (1 = small, 2 = large)
10. *area_largest* - area of the largest spot
11. *c_class_flares* - number of C-class flares production by this region in the following 24 hours (common flares), target feature
12. *m_class_flares* - number of M-class flares production by this region in the following 24 hours (moderate flares)
13. *x_class_flares* - number of X-class flares production by this region in the following 24 hours (severe flares)
====
Target Variable: c_class_flares (numeric, 8 distinct): ['0', '1', '2', '3', '4', '5', '6', '8']
====
Features:

class (nominal, 6 distinct): ['H', 'D', 'C', 'B', 'E', 'F']
largest_spot_size (nominal, 6 distinct): ['S', 'R', 'A', 'X', 'K', 'H']
spot_distribution (nominal, 4 distinct): ['O', 'X', 'I', 'C']
activity (nominal, 2 distinct): ['1', '2']
evolution (numeric, 3 distinct): ['3', '2', '1']
previous_activity (nominal, 3 distinct): ['1', '3', '2']
complex (nominal, 2 distinct): ['1', '2']
complex_path (nominal, 2 distinct): ['2', '1']
area (nominal, 2 distinct): ['1', '2']
area_largest (numeric, 1 distinct): ['1']
'''

CONTEXT = "Solar Flares"
TARGET = CuratedTarget(raw_name="c_class_flares", new_name="C-Class Flares in the following 24 Hours",
                       task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []