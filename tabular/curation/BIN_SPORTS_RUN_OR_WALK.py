from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Run_or_walk_information
====
Examples: 88588
====
URL: https://www.openml.org/search?type=data&id=40922
====
Description: **Author**: Viktor Malyi  
**Source**: [Kaggle](https://www.kaggle.com/vmalyi/run-or-walk)  
**Please cite**:   

**Run or walk**  
This dataset is gather to detect whether a person is running or walking based on deep neural networks and sensor data collected from iOS devices.

The dataset represents 88588 sensor data samples collected from the accelerometer and gyroscope from iPhone 5c in 10 seconds intervals and ~5.4/second frequency. 

### Attribute information  
This data is represented by following columns (each column contains sensor data for one of the sensor's axes):

acceleration_x
acceleration_y
acceleration_z
gyro_x
gyro_y
gyro_z

There is an activity type represented by "activity" column which acts as label and reflects following activities:

"0": walking
"1": running

The original data also contains a "wrist" column which represents the wrist where the device was placed, and "date", "time" and "username" columns which provide information about the exact date, time and user which collected these measurements.
====
Target Variable: activity (nominal, 2 distinct): ['1', '0']
====
Features:

acceleration_x (numeric, 30307 distinct): ['-0.302', '-0.2856', '-0.2926', '-0.2325', '0.1954', '-0.3022', '-0.3224', '-0.256', '-0.2914', '0.1853']
acceleration_y (numeric, 23957 distinct): ['-0.8349', '-0.7924', '-0.7917', '-0.8315', '-0.8468', '-0.8497', '-0.7971', '-0.8058', '-10.401', '-10.011']
acceleration_z (numeric, 19698 distinct): ['-0.1077', '-0.1062', '-0.1028', '-0.1095', '-0.1026', '-0.1035', '-0.1105', '-0.1066', '-0.1086', '-0.1085']
gyro_x (numeric, 40988 distinct): ['-0.6085', '0.307', '0.5811', '0.5022', '0.1116', '0.2401', '-0.2221', '0.1437', '10.221', '0.5505']
gyro_y (numeric, 38957 distinct): ['-0.3204', '-0.2268', '0.6675', '-0.1715', '-0.4009', '-0.0711', '-0.2221', '0.2358', '-0.4919', '0.2079']
gyro_z (numeric, 51296 distinct): ['-0.3787', '0.6465', '15.219', '15.181', '0.3705', '15.914', '-0.6432', '0.8791', '-0.8406', '-19.328']
'''

CONTEXT = "Sensor Data from Iphone to detect activity"
TARGET = CuratedTarget(raw_name="activity", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': "walking", '1': "running"})
COLS_TO_DROP = []
FEATURES = []