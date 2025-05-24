from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wall-robot-navigation
====
Examples: 5456
====
URL: https://www.openml.org/search?type=data&id=1497
====
Description: **Author**: Ananda Freire, Marcus Veloso and Guilherme Barreto     
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data) - 2010  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

**Wall-Following Robot Navigation Data Data Set**  
The data were collected as the SCITOS G5 robot navigates through the room following the wall in a clockwise direction, for 4 rounds, using 24 ultrasound sensors arranged circularly around its 'waist'.

The data consists of raw values of the measurements of all 24 ultrasound sensors and the corresponding class label. Sensor readings are sampled at a rate of 9 samples per second.

The class labels are:  
1. Move-Forward,  
2. Slight-Right-Turn,  
3. Sharp-Right-Turn,  
4. Slight-Left-Turn  

It is worth mentioning that the 24 ultrasound readings and the simplified distances were collected at the same time step, so each file has the same number of rows (one for each sampling time step). 

The wall-following task and data gathering were designed to test the hypothesis that this apparently simple navigation task is indeed a non-linearly separable classification task. Thus, linear classifiers, such as the Perceptron network, are not able to learn the task and command the robot around the room without collisions. Nonlinear neural classifiers, such as the MLP network, are able to learn the task and command the robot successfully without collisions. 

### Attribute Information:

1. US1: ultrasound sensor at the front of the robot (reference angle: 180°) 
2. US2: ultrasound reading (reference angle: -165°)
3. US3: ultrasound reading (reference angle: -150°)
4. US4: ultrasound reading (reference angle: -135°)
5. US5: ultrasound reading (reference angle: -120°)
6. US6: ultrasound reading (reference angle: -105°)
7. US7: ultrasound reading (reference angle: -90°)
8. US8: ultrasound reading (reference angle: -75°) 
9. US9: ultrasound reading (reference angle: -60°) 
10. US10: ultrasound reading (reference angle: -45°)
11. US11: ultrasound reading (reference angle: -30°) 
12. US12: ultrasound reading (reference angle: -15°)
13. US13: reading of ultrasound sensor situated at the back of the robot (reference angle: 0°) 
14. US14: ultrasound reading (reference angle: 15°)
15. US15: ultrasound reading (reference angle: 30°)
16. US16: ultrasound reading (reference angle: 45°)
17. US17: ultrasound reading (reference angle: 60°)
18. US18: ultrasound reading (reference angle: 75°)
19. US19: ultrasound reading (reference angle: 90°)
20. US20: ultrasound reading (reference angle: 105°)
21. US21: ultrasound reading (reference angle: 120°)
22. US22: ultrasound reading (reference angle: 135°)
23. US23: ultrasound reading (reference angle: 150°)
24. US24: ultrasound reading (reference angle: 165°)


### Relevant Papers

Ananda L. Freire, Guilherme A. Barreto, Marcus Veloso and Antonio T. Varela (2009), 'Short-Term Memory Mechanisms in Neural Network Learning of Robot Navigation Tasks: A Case Study'. Proceedings of the 6th Latin American Robotics Symposium (LARS'2009), pages 1-6
====
Target Variable: Class (nominal, 4 distinct): ['1', '2', '4', '3']
====
Features:

V1 (numeric, 1977 distinct): ['5.0', '1.373', '1.378', '1.386', '1.392', '1.779', '0.842', '1.353', '1.335', '1.346']
V2 (numeric, 2034 distinct): ['5.0', '1.414', '1.416', '1.417', '2.635', '1.411', '1.413', '1.374', '1.421', '1.302']
V3 (numeric, 1786 distinct): ['5.0', '1.414', '1.415', '1.416', '1.417', '2.611', '1.412', '1.832', '1.833', '1.424']
V4 (numeric, 1767 distinct): ['5.0', '2.627', '2.622', '2.628', '2.625', '2.623', '1.413', '1.426', '2.0', '1.455']
V5 (numeric, 1822 distinct): ['5.0', '1.448', '2.649', '1.445', '1.404', '1.924', '2.643', '1.993', '2.662', '1.925']
V6 (numeric, 1828 distinct): ['5.0', '3.244', '3.247', '3.24', '1.454', '3.206', '3.205', '3.249', '3.241', '1.909']
V7 (numeric, 1530 distinct): ['5.0', '3.241', '3.234', '3.237', '3.242', '3.267', '3.235', '3.239', '3.236', '3.233']
V8 (numeric, 2068 distinct): ['5.0', '1.325', '1.626', '1.717', '1.991', '1.525', '2.009', '2.604', '1.69', '1.483']
V9 (numeric, 1870 distinct): ['5.0', '3.275', '2.594', '3.291', '1.806', '2.715', '1.527', '1.502', '1.577', '1.578']
V10 (numeric, 2003 distinct): ['5.0', '2.592', '2.809', '2.586', '1.561', '2.594', '1.556', '2.833', '1.557', '1.538']
V11 (numeric, 1873 distinct): ['5.0', '1.557', '1.564', '2.819', '2.795', '1.724', '1.752', '0.835', '2.769', '1.582']
V12 (numeric, 1797 distinct): ['5.0', '0.823', '1.556', '0.831', '1.557', '0.826', '0.83', '0.804', '1.71', '1.724']
V13 (numeric, 1570 distinct): ['5.0', '0.808', '0.825', '1.731', '0.846', '0.836', '0.813', '1.417', '0.845', '1.71']
V14 (numeric, 1487 distinct): ['5.0', '0.78', '0.786', '0.817', '0.779', '1.522', '0.82', '0.785', '0.865', '0.832']
V15 (numeric, 1465 distinct): ['5.0', '0.792', '0.776', '0.785', '0.805', '0.851', '0.802', '0.791', '0.769', '0.806']
V16 (numeric, 1295 distinct): ['5.0', '0.754', '0.776', '0.792', '0.753', '0.75', '0.758', '0.784', '0.542', '0.763']
V17 (numeric, 1083 distinct): ['5.0', '0.493', '0.497', '0.583', '0.503', '0.506', '0.746', '0.5', '0.509', '0.502']
V18 (numeric, 971 distinct): ['5.0', '0.485', '0.493', '0.483', '0.482', '0.488', '0.487', '0.489', '0.49', '0.486']
V19 (numeric, 1042 distinct): ['5.0', '0.486', '0.483', '0.491', '0.489', '0.48', '0.482', '0.47', '0.469', '0.471']
V20 (numeric, 1136 distinct): ['5.0', '0.489', '0.491', '0.486', '0.481', '0.483', '0.488', '0.495', '0.494', '0.49']
V21 (numeric, 1355 distinct): ['5.0', '0.505', '0.514', '0.52', '0.507', '0.792', '0.803', '0.495', '0.523', '0.513']
V22 (numeric, 1736 distinct): ['5.0', '0.734', '0.737', '0.735', '0.758', '0.863', '0.768', '0.776', '0.736', '0.745']
V23 (numeric, 1758 distinct): ['5.0', '1.067', '0.856', '0.794', '1.025', '1.384', '1.367', '1.028', '1.309', '1.069']
V24 (numeric, 1856 distinct): ['5.0', '1.36', '1.083', '1.377', '0.905', '1.416', '1.358', '1.092', '1.415', '1.362']
'''

ANGLES = [180, -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

CONTEXT = "Wall Robot Navigation"
TARGET = CuratedTarget(raw_name="Class", new_name="Robot Command", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'Move-Forward',
                                      '2': 'Slight-Right-Turn',
                                      '3': 'Sharp-Right-Turn',
                                      '4': 'Slight-Left-Turn'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=f"V{i+1}", new_name=f"US{i+1} Sensor Angle {angle}")
            for i, angle in enumerate(ANGLES)]