from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: sarcos
====
Examples: 48933
====
URL: https://www.openml.org/search?type=data&id=44976
====
Description: **Data Description**

Within robotics, inverse dynamics algorithms are used to calculate the torques that a robot's motors must deliver to make the robot's end-point move in the way prescribed by its current task. More about [inverse dynamics][1].

[1]: <https://en.wikipedia.org/wiki/Inverse_dynamics>

The data set consists of 	48933 data points, collected at 100Hz from the actual robot performing various rhythmic and discrete movement tasks (this corresponds to 7.5 minutes of data collection).

Note that this version of the dataset contains both the train and test data from the original dataset, which includes many duplicates.

The task is to map from a 21-dimensional input space (7 joint positions, 7 joint velocities, 7 joint accelerations) to the corresponding 7 joint torques.

This version of the dataset includes only the training set of the original dataset, as the test dataset contains almost exclusively duplicates from the training data.

**Attribute Description**

1. *V[1-7]* - 7 joint positions
2. *V[8-14]* - 7 joint velocities
3. *V[15-21]* - 7 joint accelerations
4. *V[22-28]* - 7 joint torques, target variables, take one (*V22*) as target feature, ignore others as alternate target features
====
Target Variable: V22 (numeric, 11414 distinct): ['8.6993', '13.3208', '7.6119', '8.8352', '7.4759', '10.3304', '16.3111', '8.2915', '9.3789', '17.6704']
====
Features:

V1 (numeric, 34291 distinct): ['-0.2268', '-0.0205', '0.1256', '-0.4815', '-0.1595', '-0.1145', '-0.2721', '0.3261', '0.2411', '-0.5209']
V2 (numeric, 24862 distinct): ['-0.3174', '-0.3186', '-0.3388', '-0.3687', '-0.3189', '-0.3975', '-0.458', '-0.3816', '-0.3942', '-0.3188']
V3 (numeric, 25135 distinct): ['0.0187', '0.0075', '-0.081', '-0.1035', '-0.0562', '-0.2459', '-0.2579', '-0.0712', '-0.2579', '-0.0322']
V4 (numeric, 32846 distinct): ['1.4335', '1.7749', '1.4335', '1.8632', '1.2151', '1.3087', '1.628', '1.7037', '1.4336', '1.5689']
V5 (numeric, 24022 distinct): ['-0.0196', '0.0026', '-0.0223', '0.004', '-0.0255', '-0.0183', '0.0126', '0.012', '0.0097', '-0.0005']
V6 (numeric, 2871 distinct): ['0.1113', '0.0331', '0.0783', '0.0775', '0.0653', '0.066', '0.096', '0.1052', '0.0308', '0.1136']
V7 (numeric, 2846 distinct): ['0.2366', '0.4107', '0.2105', '0.2496', '0.1215', '0.067', '0.1775', '0.041', '0.1138', '0.1445']
V8 (numeric, 43877 distinct): ['0.0', '0.0', '0.0001', '0.0026', '-0.0019', '0.0071', '0.0031', '-0.2127', '0.0005', '-0.0019']
V9 (numeric, 42533 distinct): ['0.0', '0.0', '0.0', '0.0', '-0.0', '-0.0', '-0.0', '-0.0', '0.0', '0.0']
V10 (numeric, 43765 distinct): ['0.0', '-0.0', '-0.0', '0.0', '0.0001', '0.0001', '-0.0', '0.2997', '-0.0', '0.0001']
V11 (numeric, 44237 distinct): ['0.001', '0.2743', '-1.7038', '-0.0325', '-0.0063', '1.8863', '0.0397', '-0.0033', '1.3593', '-0.0444']
V12 (numeric, 43172 distinct): ['0.0', '0.0', '-0.0', '0.0', '-0.0', '0.0', '0.0', '-0.0003', '-0.0', '-0.0001']
V13 (numeric, 42330 distinct): ['-0.0021', '0.0015', '0.0027', '0.0002', '0.0034', '0.0022', '0.0024', '0.0016', '0.0046', '0.0031']
V14 (numeric, 44109 distinct): ['0.004', '-0.0115', '-0.0003', '0.0047', '-0.0033', '0.4158', '-0.9784', '0.0006', '-1.1444', '0.0055']
V15 (numeric, 44440 distinct): ['-1.6797', '0.2003', '-0.8813', '-1.8966', '-2.0425', '2.7052', '12.9503', '-3.4535', '5.1666', '-1.4034']
V16 (numeric, 44364 distinct): ['0.0', '-0.0', '1.7434', '0.0', '-0.0', '0.0', '0.0', '0.0013', '-0.0013', '0.0001']
V17 (numeric, 44427 distinct): ['-2.0085', '2.6959', '0.3901', '9.9025', '0.6875', '1.2658', '-6.3402', '-6.3259', '-7.2947', '2.7611']
V18 (numeric, 44457 distinct): ['-0.0419', '-17.8555', '13.5202', '-3.3467', '-8.6787', '-3.9886', '0.3806', '0.1734', '0.9128', '-1.8872']
V19 (numeric, 44394 distinct): ['0.1681', '-0.06', '-1.2097', '-0.5883', '0.0329', '-0.408', '-0.5849', '0.0313', '0.9728', '-0.0359']
V20 (numeric, 44332 distinct): ['-0.0031', '-0.052', '-0.0085', '2.255', '0.7783', '0.0943', '0.1704', '-0.0652', '0.0485', '-0.925']
V21 (numeric, 44460 distinct): ['15.5468', '-3.8418', '-0.0174', '-2.1333', '0.5338', '-0.2354', '-0.1658', '-0.6979', '-2.4626', '-22.1193']
'''

CONTEXT = "Robotics Sarcos Inverse Dynamics"
TARGET = CuratedTarget(raw_name="V22", new_name="Joint Torque", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["V23", "V24", "V25", "V26", "V27", "V28"]
FEATURES = [CuratedFeature(raw_name="V1", new_name="Joint Position 1"),
            CuratedFeature(raw_name="V2", new_name="Joint Position 2"),
            CuratedFeature(raw_name="V3", new_name="Joint Position 3"),
            CuratedFeature(raw_name="V4", new_name="Joint Position 4"),
            CuratedFeature(raw_name="V5", new_name="Joint Position 5"),
            CuratedFeature(raw_name="V6", new_name="Joint Position 6"),
            CuratedFeature(raw_name="V7", new_name="Joint Position 7"),
            CuratedFeature(raw_name="V8", new_name="Joint Velocity 1"),
            CuratedFeature(raw_name="V9", new_name="Joint Velocity 2"),
            CuratedFeature(raw_name="V10", new_name="Joint Velocity 3"),
            CuratedFeature(raw_name="V11", new_name="Joint Velocity 4"),
            CuratedFeature(raw_name="V12", new_name="Joint Velocity 5"),
            CuratedFeature(raw_name="V13", new_name="Joint Velocity 6"),
            CuratedFeature(raw_name="V14", new_name="Joint Velocity 7"),
            CuratedFeature(raw_name="V15", new_name="Joint Acceleration 1"),
            CuratedFeature(raw_name="V16", new_name="Joint Acceleration 2"),
            CuratedFeature(raw_name="V17", new_name="Joint Acceleration 3"),
            CuratedFeature(raw_name="V18", new_name="Joint Acceleration 4"),
            CuratedFeature(raw_name="V19", new_name="Joint Acceleration 5"),
            CuratedFeature(raw_name="V20", new_name="Joint Acceleration 6"),
            CuratedFeature(raw_name="V21", new_name="Joint Acceleration 7")]