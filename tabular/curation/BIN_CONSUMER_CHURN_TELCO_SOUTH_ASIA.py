from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: South_Asian_Churn_dataset
====
Examples: 2000
====
URL: https://www.openml.org/search?type=data&id=44227
====
Description: The SATO data set used is real life data collected from a major wireless telecom operator in South Asia.
====
Target Variable: Class (string, 2 distinct): ['Churned', 'Active']
====
Features:

network_age (numeric, 1364 distinct): ['117', '137', '107', '120', '122', '110', '114', '109', '112', '108']
Aggregate_Total_Rev (numeric, 1982 distinct): ['1313.1', '875.4', '437.7', '125.15', '170.8664', '1841.69', '145.33', '883.17', '1248.5104', '102.102']
Aggregate_SMS_Rev (numeric, 998 distinct): ['0.0', '1.75', '3.5', '5.25', '0.01', '7.17', '5.98', '7.0', '11.95', '9.56']
Aggregate_Data_Rev (numeric, 307 distinct): ['0.0', '1.25', '3.75', '2.5', '5.0', '6.25', '20.0', '7.5', '8.75', '22.5']
Aggregate_Data_Vol (numeric, 1990 distinct): ['64.0', '0.1211', '435072.4121', '2051331.76', '24.1104', '0.1016', '12096.7159', '105437.749', '357537.4287', '72.835']
Aggregate_Calls (numeric, 666 distinct): ['5', '2', '3', '11', '4', '1', '15', '8', '12', '6']
Aggregate_ONNET_REV (numeric, 1194 distinct): ['0', '12', '24', '228', '120', '36', '216', '502', '348', '240']
Aggregate_OFFNET_REV (numeric, 1614 distinct): ['179', '0', '358', '537', '716', '895', '1432', '1611', '1350', '609']
Aggregate_complaint_count (numeric, 20 distinct): ['1', '2', '3', '4', '5', '6', '8', '7', '11', '9']
aug_user_type (string, 4 distinct): ['3G', '2G', 'Other']
sep_user_type (string, 4 distinct): ['3G', '2G', 'Other']
aug_fav_a (string, 8 distinct): ['ptcl', 'ufone', 'mobilink', 'telenor', 'zong', 'warid', '0']
sep_fav_a (string, 7 distinct): ['ufone', 'ptcl', 'mobilink', 'telenor', 'warid', 'zong']
'''

CONTEXT = "South Asian Wireless Telcom Operator Customer Churn Prediction"
TARGET = CuratedTarget(raw_name="Class", new_name="Churn", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
