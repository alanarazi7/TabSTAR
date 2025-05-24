from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Churn_Telco_Europa
====
Examples: 190776
====
URL: https://www.openml.org/search?type=data&id=44228
====
Description: No description available
====
Target Variable: CHURN (numeric, 2 distinct): ['1', '0']
====
Features:

DAYS_LIFE (numeric, 2012 distinct): ['114', '124', '106', '211', '102', '103', '95', '100', '123', '96']
DEVICE_TECNOLOGY (numeric, 11 distinct): ['2', '1', '0', '21', '22', '4', '3', '23', '30', '24']
MIN_PLAN (numeric, 16 distinct): ['1000', '2500', '300', '400', '350', '0', '700', '150', '500', '200']
PRICE_PLAN (numeric, 38 distinct): ['13437', '16798', '8395', '21840', '10916', '25202', '3358', '6714', '2686', '5375']
TOT_MIN_CALL_OUT (numeric, 2998 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
AVG_MIN_CALL_OUT_3 (numeric, 2088 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
TOT_MIN_IN_ULT_MES (numeric, 1069 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
AVG_MIN_IN_3 (numeric, 421 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ROA_LASTMONTH (numeric, 65 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ROACETEL_LAST_MONTH (numeric, 267 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DEVICE (numeric, 18 distinct): ['13', '3', '7', '2', '5', '6', '4', '1', '10', '9']
TEC_ANT_DATA (numeric, 5 distinct): ['4.0', '3.0', '0.0', '2.5', '2.0']
STATE_DATA (numeric, 17 distinct): ['100.0', '8.0', '5.0', '9.0', '10.0', '4.0', '7.0', '6.0', '2.0', '14.0']
CITY_DATA (numeric, 336 distinct): ['345.0', '135.0', '318.0', '288.0', '163.0', '339.0', '308.0', '12.0', '240.0', '291.0']
TEC_ANT_VOICE (numeric, 5 distinct): ['4.0', '3.0', '0.0', '2.5', '2.0']
STATE_VOICE (numeric, 17 distinct): ['100.0', '8.0', '5.0', '10.0', '9.0', '4.0', '6.0', '7.0', '2.0', '14.0']
CITY_VOICE (numeric, 340 distinct): ['345.0', '135.0', '318.0', '339.0', '308.0', '288.0', '12.0', '163.0', '240.0', '291.0']
'''

CONTEXT = "Telco Europa Customers Churn Prediction"
TARGET = CuratedTarget(raw_name="Churn", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ['CNI_CUSTOMER', 'CETEL_NUMBER']
FEATURES = []
