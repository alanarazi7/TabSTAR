from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: churn
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=40701
====
Description: **Author**: Unknown  
**Source**: [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks/tree/master/datasets/classification), [BigML](https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383), Supposedly from UCI but I can't find it there.  
**Please cite**:   

A dataset relating characteristics of telephony account features and usage and whether or not the customer churned. Originally used in [Discovering Knowledge in Data: An Introduction to Data Mining](http://secs.ac.in/wp-content/CSE_PORTAL/DataMining_Daniel.pdf).
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

state (numeric, 51 distinct): ['49', '23', '1', '13', '45', '35', '43', '50', '34', '37']
account_length (numeric, 218 distinct): ['90', '87', '105', '93', '112', '101', '100', '86', '116', '103']
area_code (nominal, 3 distinct): ['415', '408', '510']
phone_number (numeric, 5000 distinct): ['2845.0', '57.0', '4530.0', '4827.0', '4463.0', '2813.0', '1454.0', '3812.0', '1936.0', '2237.0']
international_plan (nominal, 2 distinct): ['0', '1']
voice_mail_plan (nominal, 2 distinct): ['0', '1']
number_vmail_messages (numeric, 48 distinct): ['0', '31', '28', '29', '33', '24', '27', '30', '26', '32']
total_day_minutes (numeric, 1961 distinct): ['189.3', '154.0', '159.5', '180.0', '184.5', '174.5', '177.1', '183.4', '189.8', '215.6']
total_day_calls (numeric, 123 distinct): ['105', '102', '95', '94', '97', '100', '110', '112', '92', '108']
total_day_charge (numeric, 1961 distinct): ['32.18', '26.18', '27.12', '30.6', '31.37', '29.67', '30.11', '31.18', '32.27', '36.65']
total_eve_minutes (numeric, 1879 distinct): ['169.9', '199.7', '230.9', '167.6', '210.6', '216.5', '188.8', '187.5', '223.5', '194.0']
total_eve_calls (numeric, 126 distinct): ['105', '97', '91', '94', '103', '101', '96', '104', '102', '98']
total_eve_charge (numeric, 1659 distinct): ['15.9', '14.25', '16.12', '18.79', '16.97', '18.96', '19.41', '17.09', '16.8', '18.62']
total_night_minutes (numeric, 1853 distinct): ['188.2', '194.3', '186.2', '214.6', '208.9', '228.1', '210.0', '192.7', '193.6', '214.7']
total_night_calls (numeric, 131 distinct): ['105', '102', '100', '104', '99', '103', '91', '94', '95', '98']
total_night_charge (numeric, 1028 distinct): ['9.66', '8.47', '10.8', '9.63', '8.15', '9.4', '10.26', '9.45', '10.49', '10.35']
total_intl_minutes (numeric, 170 distinct): ['11.1', '9.8', '11.3', '11.4', '10.1', '10.9', '9.7', '10.6', '11.0', '10.5']
total_intl_calls (numeric, 21 distinct): ['3', '4', '2', '5', '6', '7', '1', '8', '9', '10']
total_intl_charge (numeric, 170 distinct): ['3.0', '2.65', '3.05', '3.08', '2.73', '2.94', '2.62', '2.86', '2.97', '2.84']
number_customer_service_calls (nominal, 10 distinct): ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
'''

CONTEXT = "Telephone Company Customer Churn Prediction"
TARGET = CuratedTarget(raw_name="class", new_name="Churn", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
