from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: water-treatment
====
Examples: 527
====
URL: https://www.openml.org/search?type=data&id=940
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Binarized version of the original data set (see version 1). It converts the numeric target feature to a two-class nominal target feature by computing the mean and classifying all instances with a lower target value as positive ('P') and all others as negative ('N').
====
Target Variable: binaryClass (nominal, 2 distinct): ['N', 'P']
====
Features:

ZN-E (numeric, 169 distinct): ['1.0', '1.2', '1.5', '2.0', '0.7', '3.0', '2.5', '0.8', '3.5', '0.4']
PH-E (numeric, 16 distinct): ['7.8', '7.7', '7.9', '7.6', '8.1', '7.5', '8.0', '8.2', '7.3', '7.4']
DBO-E (nominal, 205 distinct): ['133', '192', '156', '166', '179', '185', '198', '223', '159', '215']
DQO-E (nominal, 289 distinct): ['380', '392', '388', '416', '404', '400', '376', '310', '348', '321']
SS-E (nominal, 142 distinct): ['172', '186', '184', '188', '166', '208', '194', '216', '178', '204']
SSV-E (numeric, 275 distinct): ['66.7', '70.8', '71.2', '66.3', '62.0', '76.5', '55.7', '67.9', '74.0', '57.3']
SED-E (numeric, 60 distinct): ['4.5', '4.0', '3.5', '3.0', '5.5', '5.0', '2.5', '6.5', '6.0', '7.0']
COND-E (nominal, 414 distinct): ['2110', '1315', '2150', '1400', '2230', '1535', '1542', '1026', '2070', '1179']
PH-P (numeric, 13 distinct): ['7.8', '7.7', '7.9', '7.6', '8.0', '8.1', '7.5', '8.2', '8.3', '7.4']
DBO-P (nominal, 226 distinct): ['135', '182', '148', '213', '163', '161', '207', '160', '230', '117']
SS-P (nominal, 154 distinct): ['172', '236', '188', '200', '192', '156', '184', '212', '204', '218']
SSV-P (numeric, 285 distinct): ['66.7', '50.0', '63.2', '68.8', '54.2', '67.1', '68.1', '56.5', '62.3', '71.1']
SED-P (numeric, 63 distinct): ['3.0', '4.0', '4.5', '3.5', '2.5', '5.5', '5.0', '6.0', '7.0', '2.0']
COND-P (nominal, 412 distinct): ['2100', '2200', '1724', '1403', '1208', '1690', '1063', '2050', '2080', '2090']
PH-D (numeric, 13 distinct): ['7.8', '7.9', '7.7', '7.6', '8.0', '8.1', '7.5', '8.2', '7.4', '7.3']
DBO-D (nominal, 149 distinct): ['114', '135', '118', '121', '130', '97', '107', '110', '122', '119']
DQO-D (nominal, 230 distinct): ['304', '372', '274', '250', '233', '269', '204', '192', '345', '299']
SS-D (nominal, 75 distinct): ['86', '90', '74', '92', '82', '76', '104', '80', '100', '88']
SSV-D (numeric, 243 distinct): ['66.7', '80.0', '78.0', '75.0', '80.4', '77.8', '76.9', '72.2', '77.1', '72.7']
SED-D (numeric, 23 distinct): ['0.2', '0.3', '0.4', '0.5', '0.1', '0.6', '0.7', '1.0', '0.9', '0.8']
COND-D (nominal, 410 distinct): ['2130', '2140', '1529', '1356', '1401', '2190', '1155', '2240', '1188', '1420']
PH-S (numeric, 16 distinct): ['7.7', '7.8', '7.6', '7.9', '7.5', '7.4', '8.0', '7.3', '8.1', '7.1']
DBO-S (nominal, 44 distinct): ['15', '16', '18', '14', '19', '12', '17', '20', '13', '21']
DQO-S (nominal, 137 distinct): ['74', '86', '59', '75', '92', '71', '90', '78', '100', '60']
SS-S (nominal, 58 distinct): ['14', '22', '20', '18', '16', '13', '15', '19', '17', '12']
SSV-S (numeric, 193 distinct): ['80.0', '83.3', '75.0', '85.7', '77.8', '90.9', '82.9', '81.8', '90.0', '66.7']
SED-S (numeric, 18 distinct): ['0.0', '0.02', '0.01', '0.05', '0.03', '0.1', '0.2', '0.5', '0.08', '0.25']
COND-S (nominal, 413 distinct): ['1399', '1605', '1635', '1433', '1365', '1590', '2290', '1390', '1409', '1423']
RD-DBO-P (numeric, 315 distinct): ['42.3', '36.9', '41.0', '39.6', '32.8', '50.0', '50.2', '25.2', '29.1', '48.7']
RD-SS-P (numeric, 308 distinct): ['50.0', '54.5', '60.0', '55.8', '60.6', '53.2', '59.8', '61.6', '62.4', '65.5']
RD-SED-P (numeric, 144 distinct): ['93.3', '95.0', '90.0', '96.7', '80.0', '95.6', '96.4', '96.0', '94.3', '92.0']
RD-DBO-S (numeric, 185 distinct): ['85.5', '86.6', '87.0', '85.6', '87.2', '87.4', '90.8', '88.1', '86.5', '84.8']
RD-DQO-S (numeric, 265 distinct): ['73.7', '78.9', '77.4', '71.3', '66.5', '68.0', '75.0', '68.2', '75.5', '77.1']
RD-DBO-G (numeric, 156 distinct): ['90.3', '91.7', '90.2', '88.4', '88.5', '92.2', '92.1', '90.9', '89.5', '92.6']
RD-DQO-G (numeric, 230 distinct): ['82.4', '78.1', '79.5', '81.7', '75.8', '77.2', '81.0', '80.5', '84.1', '77.6']
RD-SS-G (numeric, 183 distinct): ['91.8', '91.3', '90.3', '90.7', '91.9', '93.4', '90.9', '93.2', '90.4', '92.9']
'''

CONTEXT = "Water Treatment"
TARGET = CuratedTarget(raw_name="binaryClass", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []