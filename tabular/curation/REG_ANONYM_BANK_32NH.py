from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: bank32nh
====
Examples: 8192
====
URL: https://www.openml.org/search?type=data&id=558
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

A family of datasets synthetically generated from a simulation of how bank-customers choose their banks. Tasks are
based on predicting the fraction of bank customers who leave the bank because of full queues. The bank family of
datasets are generated from a simplistic simulator, which simulates the queues in a series of banks. The simulator was
constructed with the explicit purpose of generating a family of datasets for DELVE. Customers come from several
residential areas, choose their preferred bank depending on distances and have tasks of varying complexity, and various
levels of patience. Each bank has several queues, that open and close according to demand. The tellers have various
effectivities, and customers may change queue, if their patience expires. In the rej prototasks, the object is to predict the
rate of rejections, ie the fraction of customers that are turned away from the bank because all the open tellers have full
queues.
Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at
http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html
Orginal source: DELVE repository of data.
Characteristics: Data set contains 8192 (4500+3692) cases. and 33 continuous
attributes
====
Target Variable: rej (numeric, 6284 distinct): ['0.0', '0.006', '0.0357', '0.0161', '0.0312', '0.0018', '0.0009', '0.0116', '0.0012', '0.0024']
====
Features:

a1cx (numeric, 8173 distinct): ['-0.0164', '-0.7857', '-0.484', '-0.5892', '0.5901', '-0.5471', '0.2364', '-0.0224', '0.9835', '0.0261']
a1cy (numeric, 8176 distinct): ['0.4095', '-0.256', '-0.6956', '-0.2737', '0.7257', '0.3763', '-0.1878', '-0.8275', '0.7979', '0.2856']
a1sx (numeric, 8177 distinct): ['0.0531', '1.037', '0.892', '1.3101', '0.0634', '0.1466', '0.5606', '0.5347', '1.427', '0.4425']
a1sy (numeric, 8173 distinct): ['1.0512', '0.5621', '0.2219', '0.5317', '0.4069', '0.174', '0.2043', '0.2165', '0.1761', '0.1398']
a1rho (numeric, 8162 distinct): ['0.4365', '0.2802', '-0.2303', '-0.312', '-0.024', '-0.0947', '-0.4865', '-0.2667', '0.0421', '0.4134']
a1pop (numeric, 8188 distinct): ['2.2398', '1.7289', '3.3827', '0.392', '3.2859', '0.7739', '5.0286', '2.2545', '0.091', '5.3014']
a2cx (numeric, 8176 distinct): ['0.872', '0.7775', '0.2193', '0.5289', '0.9084', '0.6288', '0.2326', '-0.3341', '0.3762', '-0.3339']
a2cy (numeric, 8175 distinct): ['-0.7191', '0.6406', '0.8987', '0.5706', '-0.1693', '-0.0623', '0.9653', '0.168', '0.3954', '0.0929']
a2sx (numeric, 8172 distinct): ['0.3726', '1.9683', '0.2106', '0.3034', '0.0898', '0.5657', '1.4894', '0.6785', '0.0459', '0.4002']
a2sy (numeric, 8174 distinct): ['0.7329', '0.1216', '0.0522', '1.0899', '0.3649', '0.9059', '0.4215', '0.1066', '0.0242', '0.2767']
a2rho (numeric, 8161 distinct): ['-0.3315', '-0.337', '0.103', '0.1122', '-0.4881', '0.483', '-0.0311', '0.1854', '-0.0229', '-0.021']
a2pop (numeric, 8185 distinct): ['1.579', '0.9798', '2.2463', '0.1867', '2.5034', '0.366', '1.3087', '2.365', '1.2397', '3.3115']
a3cx (numeric, 8173 distinct): ['0.1567', '0.8032', '-0.4978', '-0.7977', '-0.5808', '-0.3233', '-0.7692', '0.8612', '-0.6693', '0.7128']
a3cy (numeric, 8180 distinct): ['-0.0726', '0.2097', '-0.0', '-0.2127', '-0.3711', '0.0944', '0.1099', '0.5084', '0.4466', '0.2612']
a3sx (numeric, 8178 distinct): ['0.1274', '0.6267', '0.4106', '0.4916', '0.8667', '0.9464', '0.0394', '0.3277', '0.2374', '1.5486']
a3sy (numeric, 8175 distinct): ['1.1237', '0.958', '2.0305', '0.3251', '0.3411', '1.1297', '1.1361', '0.6475', '0.5416', '0.0973']
a3rho (numeric, 8161 distinct): ['-0.4745', '0.235', '-0.0838', '-0.2687', '-0.0838', '-0.1084', '-0.1977', '0.0059', '-0.3814', '0.0917']
a3pop (numeric, 8187 distinct): ['0.4441', '0.1604', '2.4857', '0.622', '0.2981', '0.6258', '0.534', '3.3273', '1.8425', '2.8528']
temp (numeric, 8159 distinct): ['0.633', '0.3581', '0.5169', '0.6573', '0.1127', '0.2812', '0.8957', '0.92', '0.553', '0.7612']
b1x (numeric, 8175 distinct): ['-0.3593', '0.2544', '0.6371', '0.7745', '0.5845', '-0.4349', '0.3642', '-0.4369', '0.725', '-0.1155']
b1y (numeric, 8170 distinct): ['0.5951', '-0.964', '-0.7452', '-0.0722', '-0.9631', '-0.4196', '-0.93', '-0.3729', '-0.8982', '0.7376']
b1call (numeric, 7 distinct): ['2', '3', '4', '5', '6', '7', '8']
b1eff (numeric, 8181 distinct): ['0.8592', '1.556', '0.6619', '1.7799', '1.3106', '1.0423', '0.5342', '0.7817', '1.7223', '0.7934']
b2x (numeric, 8175 distinct): ['0.3772', '-0.438', '0.4919', '-0.213', '-0.8981', '-0.3861', '-0.9378', '0.3196', '0.5258', '-0.1109']
b2y (numeric, 8175 distinct): ['0.751', '-0.1757', '-0.145', '0.5457', '-0.7689', '-0.9524', '-0.8728', '0.0145', '-0.6195', '0.4683']
b2call (numeric, 7 distinct): ['2', '3', '4', '5', '6', '7', '8']
b2eff (numeric, 8174 distinct): ['1.1562', '1.0957', '1.0521', '1.5133', '1.5914', '0.8625', '1.983', '1.0178', '1.605', '0.7006']
b3x (numeric, 8179 distinct): ['-0.4918', '0.4493', '-0.4155', '0.9256', '0.2889', '0.2431', '0.5706', '-0.5334', '0.4374', '0.8707']
b3y (numeric, 8173 distinct): ['-0.0848', '-0.1931', '0.8205', '0.9305', '-0.9724', '-0.3259', '-0.0503', '-0.9123', '-0.8498', '-0.2733']
b3call (numeric, 7 distinct): ['3', '2', '4', '5', '6', '7', '8']
b3eff (numeric, 8173 distinct): ['1.0869', '1.9992', '1.2301', '1.9505', '0.7465', '0.6064', '1.7423', '1.4064', '1.8037', '1.2588']
mxql (numeric, 5 distinct): ['9', '5', '7', '8', '6']
'''

CONTEXT = "Bank Customers Rejections Rate due to Full Queues"
TARGET = CuratedTarget(raw_name="rej", new_name="Rejection Rate", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []