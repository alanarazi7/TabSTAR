from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM-2_001
====
Examples: 1600
====
URL: https://www.openml.org/search?type=data&id=40650
====
Description: GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM-2_001-pmlb
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

N0 (nominal, 3 distinct): ['0', '1', '2']
N1 (nominal, 3 distinct): ['1', '0', '2']
N2 (nominal, 3 distinct): ['1', '0', '2']
N3 (nominal, 3 distinct): ['1', '0', '2']
N4 (nominal, 3 distinct): ['0', '1', '2']
N5 (nominal, 3 distinct): ['1', '0', '2']
N6 (nominal, 3 distinct): ['0', '1', '2']
N7 (nominal, 3 distinct): ['1', '0', '2']
N8 (nominal, 2 distinct): ['0', '1']
N9 (nominal, 3 distinct): ['0', '1', '2']
N10 (nominal, 3 distinct): ['0', '1', '2']
N11 (nominal, 3 distinct): ['1', '0', '2']
N12 (nominal, 3 distinct): ['0', '1', '2']
N13 (nominal, 3 distinct): ['1', '0', '2']
N14 (nominal, 3 distinct): ['1', '0', '2']
N15 (nominal, 3 distinct): ['1', '0', '2']
M0P0 (nominal, 3 distinct): ['0', '1', '2']
M0P1 (nominal, 3 distinct): ['0', '1', '2']
M1P0 (nominal, 3 distinct): ['0', '1', '2']
M1P1 (nominal, 3 distinct): ['0', '1', '2']
'''

CONTEXT = "Anonymized Gametes dataset"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []