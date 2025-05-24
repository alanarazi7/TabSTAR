from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: elevators
====
Examples: 16599
====
URL: https://www.openml.org/search?type=data&id=216
====
Description: This data set is also obtained from the task of controlling a F16
aircraft, although the target variable and attributes are different
from the ailerons domain. In this case the goal variable is related to
an action taken on the elevators of the aircraft.
====
Target Variable: Goal (numeric, 61 distinct): ['0.018', '0.019', '0.017', '0.02', '0.021', '0.016', '0.022', '0.023', '0.024', '0.015']
====
Features:

climbRate (numeric, 1500 distinct): ['-64.0', '-112.0', '52.0', '-145.0', '-103.0', '-68.0', '109.0', '-199.0', '-31.0', '-162.0']
Sgz (numeric, 180 distinct): ['-11.0', '-18.0', '-12.0', '-13.0', '-10.0', '-15.0', '-9.0', '-16.0', '-14.0', '-7.0']
p (numeric, 202 distinct): ['0.12', '0.02', '0.01', '0.22', '0.14', '0.13', '-0.03', '-0.02', '0.06', '0.03']
q (numeric, 100 distinct): ['0.04', '0.07', '0.02', '0.05', '0.03', '0.08', '0.01', '0.06', '-0.02', '0.09']
curRoll (numeric, 60 distinct): ['0.3', '0.6', '0.4', '0.7', '0.2', '-0.4', '-0.5', '-0.7', '0.1', '-0.3']
absRoll (numeric, 21 distinct): ['-7.0', '-6.0', '-11.0', '-10.0', '-8.0', '-12.0', '-13.0', '-9.0', '-14.0', '-5.0']
diffClb (numeric, 91 distinct): ['1.0', '-2.0', '-3.0', '2.0', '3.0', '4.0', '-4.0', '-6.0', '5.0', '-5.0']
diffRollRate (numeric, 113 distinct): ['-0.002', '0.001', '-0.004', '-0.006', '0.004', '-0.003', '0.002', '0.003', '-0.005', '0.005']
diffDiffClb (numeric, 134 distinct): ['-0.1', '0.0', '-0.2', '0.1', '-0.3', '0.2', '0.3', '-0.4', '-0.5', '0.4']
SaTime1 (numeric, 35 distinct): ['-0.0004', '-0.0005', '-0.0007', '-0.0006', '-0.0008', '-0.0009', '-0.0003', '-0.001', '-0.0011', '-0.0012']
SaTime2 (numeric, 35 distinct): ['-0.0005', '-0.0004', '-0.0006', '-0.0007', '-0.0003', '-0.0008', '-0.0009', '-0.001', '-0.0011', '-0.0012']
SaTime3 (numeric, 35 distinct): ['-0.0005', '-0.0004', '-0.0006', '-0.0007', '-0.0003', '-0.0008', '-0.0009', '-0.001', '-0.0011', '-0.0012']
SaTime4 (numeric, 34 distinct): ['-0.0005', '-0.0004', '-0.0006', '-0.0007', '-0.0003', '-0.0008', '-0.0009', '-0.001', '-0.0011', '-0.0012']
diffSaTime1 (numeric, 15 distinct): ['0.0', '0.0001', '-0.0002', '-0.0001', '0.0002', '-0.0003', '0.0003', '-0.0004', '0.0004', '-0.0005']
diffSaTime2 (numeric, 3 distinct): ['0.0', '0.0001', '0.0002']
diffSaTime3 (numeric, 12 distinct): ['0.0', '0.0001', '-0.0002', '-0.0001', '0.0002', '-0.0003', '0.0003', '-0.0004', '0.0004', '-0.0005']
diffSaTime4 (numeric, 3 distinct): ['0.0', '0.0002', '0.0001']
Sa (numeric, 34 distinct): ['-0.0005', '-0.0004', '-0.0006', '-0.0007', '-0.0003', '-0.0008', '-0.0009', '-0.001', '-0.0011', '-0.0012']
'''

CONTEXT = "F16 Flight Simulation Elevators"
TARGET = CuratedTarget(raw_name="Goal", new_name="Action related to elevators of the Aircraft",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []