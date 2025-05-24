from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mobile_churn
====
Examples: 66469
====
URL: https://www.openml.org/search?type=data&id=44231
====
Description: No description available
====
Target Variable: churn (numeric, 2 distinct): ['0', '1']
====
Features:

year (numeric, 1 distinct): ['2013']
month (numeric, 3 distinct): ['6', '8', '7']
user_account_id (numeric, 66469 distinct): ['13', '979888', '979601', '979606', '979640', '979647', '979677', '979681', '979695', '979720']
user_lifetime (numeric, 2057 distinct): ['15947', '15885', '1986', '1985', '1984', '15880', '1983', '1982', '1981', '1980']
user_intake (numeric, 2 distinct): ['0', '1']
user_no_outgoing_activity_in_days (numeric, 100 distinct): ['1', '2', '8', '3', '1276', '4', '5', '6', '7', '9']
user_account_balance_last (numeric, 5512 distinct): ['0.0', '0.01', '0.02', '0.04', '0.03', '0.05', '0.06', '15.0', '7.5', '0.07']
user_spendings (numeric, 5014 distinct): ['0.0', '0.15', '0.3', '0.06', '0.45', '0.18', '0.6', '0.36', '0.12', '0.9']
user_has_outgoing_calls (numeric, 2 distinct): ['1', '0']
user_has_outgoing_sms (numeric, 2 distinct): ['1', '0']
user_use_gprs (numeric, 2 distinct): ['0', '1']
user_does_reload (numeric, 2 distinct): ['1', '0']
reloads_inactive_days (numeric, 99 distinct): ['1276', '8', '16', '1', '2', '27', '3', '4', '5', '6']
reloads_count (numeric, 25 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
reloads_sum (numeric, 1222 distinct): ['0.0', '12.0', '6.0', '5.0', '24.02', '1.5', '15.0', '4.0', '6.01', '36.03']
calls_outgoing_count (numeric, 582 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
calls_outgoing_spendings (numeric, 4241 distinct): ['0.0', '0.15', '0.3', '0.45', '0.6', '0.75', '0.9', '0.18', '1.05', '1.2']
calls_outgoing_duration (numeric, 10051 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '0.02']
calls_outgoing_spendings_max (numeric, 1130 distinct): ['0.0', '0.15', '0.18', '0.25', '0.3', '0.2', '0.29', '0.49', '2.0', '0.27']
calls_outgoing_duration_max (numeric, 2427 distinct): ['0.0', '1.0', '2.0', '3.0', '0.78', '60.0', '5.0', '4.0', '1.08', '1.02']
calls_outgoing_inactive_days (numeric, 102 distinct): ['1', '2', '1338', '3', '4', '1276', '5', '6', '8', '9']
calls_outgoing_to_onnet_count (numeric, 66 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
calls_outgoing_to_onnet_spendings (numeric, 863 distinct): ['0.0', '0.18', '0.35', '0.53', '0.41', '0.19', '1.85', '0.58', '0.38', '0.2']
calls_outgoing_to_onnet_duration (numeric, 1075 distinct): ['0.0', '0.5', '1.0', '1.5', '2.0', '1.18', '0.75', '1.65', '0.53', '0.57']
calls_outgoing_to_onnet_inactive_days (numeric, 102 distinct): ['1', '2', '1338', '3', '4', '1276', '5', '6', '8', '9']
calls_outgoing_to_offnet_count (numeric, 367 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
calls_outgoing_to_offnet_spendings (numeric, 3121 distinct): ['0.0', '0.15', '0.3', '0.45', '0.6', '0.75', '0.9', '0.18', '1.05', '1.2']
calls_outgoing_to_offnet_duration (numeric, 7967 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
calls_outgoing_to_offnet_inactive_days (numeric, 102 distinct): ['1', '2', '1338', '3', '4', '1276', '5', '6', '8', '9']
calls_outgoing_to_abroad_count (numeric, 160 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
calls_outgoing_to_abroad_spendings (numeric, 1208 distinct): ['0.0', '0.18', '0.04', '0.03', '0.05', '0.02', '0.35', '0.08', '0.07', '0.2']
calls_outgoing_to_abroad_duration (numeric, 1837 distinct): ['0.0', '0.5', '1.0', '2.0', '1.5', '0.53', '1.08', '0.6', '0.78', '0.75']
calls_outgoing_to_abroad_inactive_days (numeric, 102 distinct): ['1', '2', '1338', '3', '4', '1276', '5', '6', '8', '9']
sms_outgoing_count (numeric, 807 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sms_outgoing_spendings (numeric, 2253 distinct): ['0.0', '0.06', '0.12', '0.18', '0.24', '0.3', '0.36', '0.42', '0.48', '0.54']
sms_outgoing_spendings_max (numeric, 22 distinct): ['0.0', '0.06', '0.11', '0.25', '0.09', '0.15', '0.55', '1.0', '0.5', '2.0']
sms_outgoing_inactive_days (numeric, 101 distinct): ['1', '1276', '1338', '2', '3', '4', '5', '6', '9', '7']
sms_outgoing_to_onnet_count (numeric, 209 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sms_outgoing_to_onnet_spendings (numeric, 202 distinct): ['0.0', '0.06', '0.12', '0.18', '0.24', '0.3', '0.36', '0.42', '0.48', '0.54']
sms_outgoing_to_onnet_inactive_days (numeric, 101 distinct): ['1', '1276', '1338', '2', '3', '4', '5', '6', '9', '7']
sms_outgoing_to_offnet_count (numeric, 645 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sms_outgoing_to_offnet_spendings (numeric, 700 distinct): ['0.0', '0.06', '0.12', '0.18', '0.24', '0.3', '0.36', '0.42', '0.54', '0.48']
sms_outgoing_to_offnet_inactive_days (numeric, 101 distinct): ['1', '1276', '1338', '2', '3', '4', '5', '6', '9', '7']
sms_outgoing_to_abroad_count (numeric, 149 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sms_outgoing_to_abroad_spendings (numeric, 237 distinct): ['0.0', '0.11', '0.22', '0.33', '0.44', '0.55', '0.66', '0.77', '0.88', '1.1']
sms_outgoing_to_abroad_inactive_days (numeric, 101 distinct): ['1', '1276', '1338', '2', '3', '4', '5', '6', '9', '7']
sms_incoming_count (numeric, 191 distinct): ['0', '3', '4', '7', '8', '5', '9', '6', '10', '12']
sms_incoming_spendings (numeric, 161 distinct): ['0.0', '2.0', '1.0', '1.3', '4.0', '3.0', '1.5', '0.15', '2.6', '6.0']
sms_incoming_from_abroad_count (numeric, 71 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sms_incoming_from_abroad_spendings (numeric, 18 distinct): ['0.0', '2.0', '1.0', '1.5', '10.0', '4.0', '0.5', '2.6', '3.0', '0.3']
gprs_session_count (numeric, 558 distinct): ['0', '3', '6', '9', '12', '15', '18', '21', '4', '8']
gprs_usage (numeric, 1526 distinct): ['0.0', '0.3', '0.31', '0.59', '0.32', '0.6', '0.33', '0.61', '0.62', '0.35']
gprs_spendings (numeric, 514 distinct): ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.35', '0.3', '0.4', '0.45']
gprs_inactive_days (numeric, 101 distinct): ['1276', '1338', '1', '1307', '1271', '2', '3', '9', '4', '5']
last_100_reloads_count (numeric, 105 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
last_100_reloads_sum (numeric, 3570 distinct): ['0.0', '15.0', '12.0', '6.0', '24.0', '4.0', '10.0', '24.015', '36.0', '18.0']
last_100_calls_outgoing_duration (numeric, 18236 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '6.0', '5.0', '7.0', '9.0', '8.0']
last_100_calls_outgoing_to_onnet_duration (numeric, 1707 distinct): ['0.0', '0.5', '1.0', '1.5', '2.0', '0.53', '0.57', '1.18', '0.52', '0.8']
last_100_calls_outgoing_to_offnet_duration (numeric, 14743 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
last_100_calls_outgoing_to_abroad_duration (numeric, 2915 distinct): ['0.0', '0.5', '1.0', '2.0', '0.53', '1.5', '1.08', '1.38', '0.72', '0.33']
last_100_sms_outgoing_count (numeric, 1611 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
last_100_sms_outgoing_to_onnet_count (numeric, 401 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
last_100_sms_outgoing_to_offnet_count (numeric, 1243 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7', '9']
last_100_sms_outgoing_to_abroad_count (numeric, 244 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
last_100_gprs_usage (numeric, 2630 distinct): ['0.0', '0.3', '0.31', '0.59', '0.32', '0.6', '0.61', '0.33', '0.62', '0.35']
'''

CONTEXT = "Mobile Operator Customer Churn Prediction"
TARGET = CuratedTarget(raw_name="churn", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["year", "user_account_id"]
FEATURES = []
