from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: walking-activity
====
Examples: 149332
====
URL: https://www.openml.org/search?type=data&id=1509
====
Description: **Author**: P. Casale, O. Pujol, P. Radeva.    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity)  
**Please cite**: Casale, P. Pujol, O. and Radeva, P. 'Personalization and user verification in wearable systems using biometric walking patterns' Personal and Ubiquitous Computing, 16(5), 563-580, 2012

**User Identification From Walking Activity Data Set**  
The dataset collects data from an Android smartphone positioned in the chest pocket. Accelerometer Data are collected from 22 participants walking in the wild over a predefined path. The dataset is intended for Activity Recognition research purposes. It provides challenges for identification and authentication of people using motion patterns. 

**Note: the original per-user datasets were joined into one dataset**

### Attribute Information  
Time-step, x acceleration, y acceleration, z acceleration  
Target: User ID.
====
Target Variable: Class (nominal, 22 distinct): ['21', '22', '4', '18', '6', '15', '9', '17', '12', '3']
====
Features:

V1 (numeric, 72087 distinct): ['0.0', '20.17', '20.02', '27.4', '19.51', '58.2', '58.14', '118.98', '13.78', '24.69']
V2 (numeric, 756 distinct): ['-0.6946', '-0.5312', '-0.6538', '-0.9126', '-1.076', '-0.8853', '-0.8036', '-0.572', '-0.7627', '-0.504']
V3 (numeric, 665 distinct): ['8.8532', '8.8124', '8.7306', '8.7715', '8.9213', '8.6898', '8.9622', '9.5751', '8.6625', '9.112']
V4 (numeric, 733 distinct): ['-0.3405', '-0.3814', '-0.6538', '-0.2997', '-0.504', '-0.5312', '-0.4222', '-0.1907', '-0.2316', '-0.6129']
'''

CONTEXT = "Walking Activity of Users collected from Android Smartphone"
TARGET = CuratedTarget(raw_name="Class", new_name="User ID", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Time Step"),
            CuratedFeature(raw_name="V2", new_name="X Acceleration"),
            CuratedFeature(raw_name="V3", new_name="Y Acceleration"),
            CuratedFeature(raw_name="V4", new_name="Z Acceleration")]