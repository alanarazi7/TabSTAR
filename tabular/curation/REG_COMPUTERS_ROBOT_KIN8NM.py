from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: kin8nm
====
Examples: 8192
====
URL: https://www.openml.org/search?type=data&id=44980
====
Description: **Data Description**

A realistic simulation of the forward dynamics of an 8 link all-revolute robot arm. The task in all datasets is to predict the distance of the end-effector from a target. The input are the angular positions of the joints. The task is medium noisy and nonlinear.

Each instance represents a configuration of angular positions of the joints and the resulting distance of the end-effector from a target.

**Attribute Description**

1. *theta[1-8]* - angular positions of the joints
2. *y* - resulting distance of end-effector from target, target feature
====
Target Variable: y (numeric, 8191 distinct): ['0.5349', '0.5365', '0.613', '0.3365', '0.972', '0.9505', '0.3945', '0.7194', '0.8265', '1.1248']
====
Features:

theta1 (numeric, 8192 distinct): ['-0.0151', '0.3605', '-1.2933', '0.5069', '0.8851', '0.4221', '-0.9901', '-0.0053', '0.3243', '-1.2879']
theta2 (numeric, 8192 distinct): ['0.3607', '-0.3014', '0.5197', '-0.6865', '-0.4711', '-1.2638', '0.224', '0.8544', '-0.5422', '-0.7578']
theta3 (numeric, 8192 distinct): ['0.4694', '0.6292', '0.0724', '-0.7669', '-0.3414', '1.0039', '-1.357', '-0.075', '-1.0819', '-1.5562']
theta4 (numeric, 8190 distinct): ['-0.1288', '1.3715', '1.3097', '0.8129', '-0.8898', '-1.4388', '-0.3294', '-0.5471', '-0.4218', '-0.9909']
theta5 (numeric, 8192 distinct): ['0.988', '-0.7416', '-1.2814', '0.2859', '0.8685', '0.8111', '0.7829', '-0.1553', '0.317', '0.563']
theta6 (numeric, 8191 distinct): ['-1.1309', '-0.0255', '-0.0809', '1.1431', '-0.5553', '-0.3098', '-0.9184', '0.8327', '-1.0256', '-1.0162']
theta7 (numeric, 8192 distinct): ['0.6641', '-1.0384', '-0.4816', '0.4762', '0.7642', '-1.0979', '1.1742', '0.9937', '0.5788', '-0.3488']
theta8 (numeric, 8192 distinct): ['0.0628', '-0.7175', '-0.7673', '1.4802', '0.1165', '1.4253', '-0.1726', '-1.0539', '-1.4167', '-0.1978']
'''

CONTEXT = "Simulation of Forward Dynamics of an 8 Link Robot Arm"
TARGET = CuratedTarget(raw_name="y", new_name="Distance from Target", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []