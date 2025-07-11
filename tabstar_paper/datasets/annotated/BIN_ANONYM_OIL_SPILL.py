from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: oil_spill
====
Examples: 937
====
URL: https://www.openml.org/search?type=data&id=311
====
Description: **Author**:   
  
**Source**: Unknown -   
**Please cite**:   

Oil dataset

Past Usage:
1. Kubat, M., Holte, R.,
====
Target Variable: class (nominal, 2 distinct): ['-1', '1']
====
Features:

attr1 (numeric, 238 distinct): ['3.0', '5.0', '6.0', '8.0', '9.0', '11.0', '2.0', '1.0', '52.0', '23.0']
attr2 (numeric, 297 distinct): ['10.0', '12.0', '11.0', '15.0', '13.0', '51.0', '14.0', '18.0', '16.0', '19.0']
attr3 (numeric, 927 distinct): ['112.12', '22.0', '7.7', '118.31', '26.5', '3.0', '4.49', '18.25', '133.64', '34.54']
attr4 (numeric, 933 distinct): ['1054.81', '644.0', '1824.79', '283.29', '456.63', '587.4', '148.4', '540.47', '108.27', '607.96']
attr5 (numeric, 179 distinct): ['90', '69', '66', '64', '68', '73', '74', '70', '72', '65']
attr6 (numeric, 375 distinct): ['81000.0', '97200.0', '89100.0', '121500.0', '105300.0', '113400.0', '127500.0', '145800.0', '129600.0', '153900.0']
attr7 (numeric, 820 distinct): ['37.33', '38.9', '54.5', '55.0', '38.57', '39.82', '54.89', '26.72', '30.63', '42.08']
attr8 (numeric, 618 distinct): ['7.68', '8.79', '7.73', '7.59', '6.47', '7.8', '9.37', '7.44', '8.16', '6.88']
attr9 (numeric, 561 distinct): ['831.0', '794.0', '1191.0', '921.0', '974.0', '1011.0', '1461.0', '884.0', '1101.0', '847.0']
attr10 (numeric, 57 distinct): ['0.2', '0.17', '0.18', '0.19', '0.21', '0.16', '0.22', '0.23', '0.15', '0.14']
attr11 (numeric, 577 distinct): ['102.0', '97.5', '110.4', '110.0', '107.2', '95.6', '99.8', '88.4', '85.7', '105.5']
attr12 (numeric, 59 distinct): ['0.21', '0.23', '0.24', '0.2', '0.25', '0.18', '0.19', '0.22', '0.27', '0.26']
attr13 (numeric, 73 distinct): ['0.26', '0.28', '0.3', '0.31', '0.24', '0.23', '0.29', '0.25', '0.27', '0.34']
attr14 (numeric, 107 distinct): ['0.34', '0.36', '0.35', '0.33', '0.3', '0.37', '0.39', '0.49', '0.38', '0.41']
attr15 (numeric, 53 distinct): ['0.19', '0.2', '0.17', '0.15', '0.18', '0.12', '0.14', '0.16', '0.11', '0.13']
attr16 (numeric, 91 distinct): ['0.11', '0.09', '0.1', '0.08', '0.16', '0.12', '0.07', '0.06', '0.13', '0.15']
attr17 (numeric, 893 distinct): ['22.82', '22.6', '22.42', '41.65', '71.21', '22.55', '20.82', '99.46', '21.78', '35.52']
attr18 (numeric, 810 distinct): ['11.61', '8.31', '6.76', '14.09', '12.51', '17.25', '10.14', '4.95', '12.78', '12.15']
attr19 (numeric, 170 distinct): ['0.36', '0.37', '0.35', '0.34', '0.38', '0.33', '0.39', '0.4', '0.93', '0.9']
attr20 (numeric, 53 distinct): ['0.24', '0.23', '0.18', '0.17', '0.2', '0.19', '0.21', '0.25', '0.22', '0.16']
attr21 (numeric, 68 distinct): ['0.27', '0.24', '0.28', '0.29', '0.3', '0.21', '0.25', '0.31', '0.23', '0.26']
attr22 (numeric, 9 distinct): ['55.85', '75.26', '85.22', '67.87', '69.09', '47.66', '123.47', '126.08', '87.65']
attr23 (numeric, 1 distinct): ['0']
attr24 (numeric, 92 distinct): ['0.46', '0.47', '0.48', '0.45', '0.49', '0.44', '0.5', '1.01', '0.52', '1.07']
attr25 (numeric, 9 distinct): ['221.97', '351.67', '422.12', '421.21', '239.69', '204.34', '2036.8', '2025.42', '132.78']
attr26 (numeric, 8 distinct): ['0.18', '0.87', '1.01', '1.83', '0.97', '-0.71', '-0.53', '-0.01']
attr27 (numeric, 9 distinct): ['5.07', '9.24', '12.06', '14.78', '3.83', '4.66', '2.96', '3.01', '3.78']
attr28 (numeric, 308 distinct): ['0.21', '0.26', '0.24', '0.17', '0.04', '0.22', '0.59', '0.32', '0.08', '0.19']
attr29 (numeric, 447 distinct): ['2.27', '2.67', '2.89', '3.6', '3.41', '4.39', '2.57', '3.36', '1.95', '2.6']
attr30 (numeric, 392 distinct): ['-2.98', '-3.0', '-2.91', '-2.97', '-2.96', '-3.11', '-0.83', '-2.94', '-2.92', '-3.07']
attr31 (numeric, 107 distinct): ['-0.33', '-0.36', '-0.41', '-0.25', '-0.34', '-0.38', '-0.29', '-0.31', '-0.45', '-0.37']
attr32 (numeric, 42 distinct): ['1.09', '2.17', '1.22', '2.16', '1.94', '1.1', '2.91', '2.18', '1.95', '2.92']
attr33 (numeric, 4 distinct): ['0.0', '0.87', '0.01', '0.86']
attr34 (numeric, 45 distinct): ['1.09', '2.17', '1.22', '2.16', '1.94', '1.1', '2.91', '2.18', '1.95', '2.92']
attr35 (numeric, 141 distinct): ['8.0', '9.0', '7.0', '10.0', '6.0', '12.0', '13.0', '20.0', '22.0', '27.0']
attr36 (numeric, 110 distinct): ['630.0', '540.0', '450.0', '810.0', '720.0', '1530.0', '1170.0', '900.0', '990.0', '1080.0']
attr37 (numeric, 3 distinct): ['0.01', '0.0', '0.02']
attr38 (numeric, 758 distinct): ['7.78', '7.75', '8.53', '11.67', '9.71', '14.32', '9.98', '7.07', '12.47', '8.73']
attr39 (numeric, 9 distinct): ['82', '78', '85', '64', '102', '99', '143', '133', '89']
attr40 (numeric, 9 distinct): ['50', '55', '63', '39', '73', '67', '86', '85', '69']
attr41 (numeric, 388 distinct): ['0.0', '402.49', '484.66', '569.21', '254.56', '360.0', '649.0', '381.84', '450.0', '524.79']
attr42 (numeric, 220 distinct): ['180.0', '0.0', '127.28', '90.0', '254.56', '270.0', '360.56', '316.23', '269.26', '250.0']
attr43 (numeric, 644 distinct): ['0.0', '90.0', '63.64', '135.0', '180.0', '127.28', '126.0', '112.5', '84.85', '150.0']
attr44 (numeric, 649 distinct): ['0.0', '73.48', '51.96', '63.64', '49.3', '90.0', '56.92', '40.25', '45.0', '68.03']
attr45 (numeric, 499 distinct): ['0.0', '6.32', '4.47', '4.0', '2.0', '2.67', '3.58', '4.12', '2.37', '3.85']
attr46 (numeric, 2 distinct): ['0', '1']
attr47 (numeric, 937 distinct): ['33243.19', '5640.18', '5427.85', '4536.06', '14901.36', '6439.07', '16498.12', '4189.15', '15308.08', '6026.71']
attr48 (numeric, 169 distinct): ['65.87', '65.95', '66.17', '65.94', '65.93', '65.92', '66.16', '65.89', '65.88', '65.86']
attr49 (numeric, 286 distinct): ['7.22', '6.34', '7.84', '7.71', '6.27', '6.14', '14.92', '6.26', '6.3', '6.2']
'''

CONTEXT = "Anonymized: Oil Spill Dataset"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []