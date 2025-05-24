from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Data_Science_Nigeria_Telecoms_Churn
====
Examples: 1400
====
URL: https://www.openml.org/search?type=data&id=44230
====
Description: Churn happens when customers are leaving their current service provider and moving to another one. This is a big business problem because it is more expensive to acquire a new customer than to keep an existing one from leaving. This explains this challenge, which is one of the pre-qualification to the 2018 Data Science Nigeria all-expense paid bootcam/hackathon scheduled to hold 10-15 October, 2018. Read up, fine your model, build incremental statistical thinking and reinforce your Data Science career growth. An academic data has been used here to help participants classify potential customers who might churn. Please note that a balanced dataset has been used
====
Target Variable: churn_status (numeric, 2 distinct): ['0.0', '1.0']
====
Features:

customer_id (string, 1400 distinct): ['ADF0039', 'ADF1977', 'ADF1997', 'ADF1994', 'ADF1993', 'ADF1991', 'ADF1990', 'ADF1988', 'ADF1987', 'ADF1986']
network_age (numeric, 1038 distinct): ['117.0', '107.0', '120.0', '110.0', '109.0', '112.0', '123.0', '133.0', '108.0', '137.0']
customer_tenure_in_month (numeric, 1038 distinct): ['3.9', '3.57', '4.0', '3.67', '3.63', '3.73', '4.1', '4.43', '3.6', '4.57']
total_spend_in_months_1_and_2_of_2017 (numeric, 1387 distinct): ['875.4', '437.7', '125.15', '1841.69', '883.17', '170.8664', '145.33', '1248.5104', '219.0592', '1659.2812']
total_sms_spend (numeric, 712 distinct): ['0.0', '1.75', '3.5', '5.25', '0.01', '7.17', '9.56', '7.0', '11.95', '5.98']
total_data_spend (numeric, 241 distinct): ['0.0', '1.25', '2.5', '3.75', '5.0', '6.25', '20.0', '7.5', '8.75', '22.5']
total_data_consumption (numeric, 1394 distinct): ['64.0', '435072.4121', '12096.7159', '0.1016', '2051331.76', '2024585.18', '191.1318', '1881350.401', '36.0', '107.4082']
total_unique_calls (numeric, 503 distinct): ['2.0', '5.0', '3.0', '4.0', '1.0', '11.0', '8.0', '15.0', '12.0', '16.0']
total_onnet_spend_ (numeric, 851 distinct): ['0.0', '24.0', '12.0', '228.0', '120.0', '348.0', '36.0', '60.0', '216.0', '48.0']
total_offnet_spend (numeric, 1126 distinct): ['179.0', '0.0', '358.0', '537.0', '716.0', '895.0', '1432.0', '609.0', '395.0', '228.0']
total_call_centre_complaint_calls (numeric, 19 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '11.0', '9.0']
network_type_subscription_in_month_1 (string, 4 distinct): ['3G', '2G', 'Other']
network_type_subscription_in_month_2 (string, 4 distinct): ['3G', 'Other', '2G']
most_loved_competitor_network_in_in_month_1 (string, 8 distinct): ['PQza', 'Uxaa', 'Mango', 'ToCall', 'Zintel', 'Weematel', '0']
most_loved_competitor_network_in_in_month_2 (string, 7 distinct): ['Uxaa', 'PQza', 'Mango', 'ToCall', 'Weematel', 'Zintel']
'''

CONTEXT = "Nigerian Telcom Company Customer Churn Prediction"
TARGET = CuratedTarget(raw_name="churn_status", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["customer_id"]
FEATURES = []
