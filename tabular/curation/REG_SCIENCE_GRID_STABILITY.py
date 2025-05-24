from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: grid_stability
====
Examples: 10000
====
URL: https://www.openml.org/search?type=data&id=44973
====
Description: **Data Description**

The local stability analysis of the 4-node star system (electricity producer is in the center) implementing Decentral Smart Grid Control concept was performed.

This dataset contains simulations regarding electrical grid stability. The model is composed of a generator model and an economic model.

The analysis is performed for different sets of input values. Several input values are kept the same: averaging time - 2s, coupling strength - 8s^-2, damping - 0.1s^-1.

The goal is to estimate the stability of the system.

**Attribute Description**

14 features describing the system:

1. *tau[1-4]* - reaction time of participant (real from the range [0.5,10]s), tau1 - value for electricity producer
2. *p[1-4]* - nominal power consumed(negative) / produced(positive)(real). For consumers from the range [-0.5,-2]s^-2; p1 = abs(p2 + p3 + p4)
3. *g[1-4]* - coefficient (gamma) proportional to price elasticity (real from the range [0.05,1]s^-1), g1 - the value for electricity producer
4. *stab* - the maximal real part of the characteristic equation root (if positive - the system is linearly unstable), target feature
5. *stabf* - the stability label of the system (categorical: stable/unstable), alternate target feature for a classification task
====
Target Variable: stab (numeric, 10000 distinct): ['0.0553', '0.0078', '0.0533', '0.0606', '0.0345', '0.0609', '-0.0466', '0.0804', '0.0505', '0.0187']
====
Features:

tau1 (numeric, 10000 distinct): ['2.9591', '4.9281', '8.9423', '9.739', '5.5938', '4.7457', '1.2613', '8.5592', '7.5813', '6.5896']
tau2 (numeric, 10000 distinct): ['3.0799', '5.4019', '3.2354', '8.4862', '8.0717', '4.1571', '4.7927', '8.3865', '4.0613', '7.4377']
tau3 (numeric, 10000 distinct): ['8.381', '4.5767', '8.1372', '7.8162', '5.3473', '7.856', '1.5183', '4.1308', '6.1926', '1.9801']
tau4 (numeric, 10000 distinct): ['9.7808', '1.7214', '7.0459', '9.3808', '6.0721', '7.4786', '4.0552', '4.2214', '6.5573', '3.198']
p1 (numeric, 10000 distinct): ['3.7631', '4.7707', '4.2475', '3.8707', '2.8433', '2.8481', '4.3689', '2.4164', '3.2212', '4.9185']
p2 (numeric, 10000 distinct): ['-0.7826', '-1.3952', '-0.6636', '-1.4271', '-0.9995', '-1.5826', '-1.9806', '-1.1435', '-1.4336', '-1.2646']
p3 (numeric, 10000 distinct): ['-1.2574', '-1.6462', '-1.9868', '-0.5966', '-0.9335', '-0.6694', '-1.3922', '-0.712', '-0.5911', '-1.7927']
p4 (numeric, 10000 distinct): ['-1.7231', '-1.7292', '-1.5971', '-1.8471', '-0.9102', '-0.596', '-0.9961', '-0.5609', '-1.1966', '-1.8613']
g1 (numeric, 10000 distinct): ['0.6505', '0.691', '0.8129', '0.1792', '0.6267', '0.6484', '0.104', '0.9278', '0.2171', '0.7001']
g2 (numeric, 10000 distinct): ['0.8596', '0.5283', '0.9296', '0.4019', '0.113', '0.5949', '0.0544', '0.8053', '0.6806', '0.5628']
g3 (numeric, 10000 distinct): ['0.8874', '0.148', '0.0701', '0.8431', '0.051', '0.375', '0.6699', '0.838', '0.4573', '0.8279']
g4 (numeric, 10000 distinct): ['0.958', '0.355', '0.774', '0.6223', '0.8862', '0.7663', '0.2505', '0.5941', '0.7263', '0.1504']
'''

CONTEXT = "Grid Stability Analysis"
TARGET = CuratedTarget(raw_name="stab", new_name="Stability", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["stabf"]
FEATURES = [CuratedFeature(raw_name="tau1", new_name="Reaction Time 1"),
            CuratedFeature(raw_name="tau2", new_name="Reaction Time 2"),
            CuratedFeature(raw_name="tau3", new_name="Reaction Time 3"),
            CuratedFeature(raw_name="tau4", new_name="Reaction Time 4"),
            CuratedFeature(raw_name="p1", new_name="Nominal Power 1"),
            CuratedFeature(raw_name="p2", new_name="Nominal Power 2"),
            CuratedFeature(raw_name="p3", new_name="Nominal Power 3"),
            CuratedFeature(raw_name="p4", new_name="Nominal Power 4"),
            CuratedFeature(raw_name="g1", new_name="Coefficient 1"),
            CuratedFeature(raw_name="g2", new_name="Coefficient 2"),
            CuratedFeature(raw_name="g3", new_name="Coefficient 3"),
            CuratedFeature(raw_name="g4", new_name="Coefficient 4")]