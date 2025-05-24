from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Fraud-Detection-Updated
====
Examples: 4156
====
URL: https://www.openml.org/search?type=data&id=46359
====
Description: Updated Fraud Detection dataset with nominal target for binary classification.
====
Target Variable: bad_flag (nominal, 2 distinct): ['0', '1']
====
Features:

dpd_5_cnt (numeric, 9 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '0.0', '6.0', '7.0']
dpd_15_cnt (numeric, 7 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0']
dpd_30_cnt (numeric, 5 distinct): ['0.0', '1.0', '2.0', '3.0']
close_loans_cnt (numeric, 22 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
federal_district_nm (numeric, 9 distinct): ['3', '6', '2', '4', '1', '0', '5', '7', '-1']
payment_type_0 (numeric, 9 distinct): ['0', '1', '2', '3', '5', '4', '15', '6', '8']
payment_type_1 (numeric, 28 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
payment_type_2 (numeric, 26 distinct): ['0', '3', '1', '2', '6', '4', '5', '7', '8', '9']
payment_type_3 (numeric, 24 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '7', '9']
payment_type_4 (numeric, 7 distinct): ['0', '1', '2', '3', '5', '4', '7']
payment_type_5 (numeric, 1 distinct): ['0']
past_billings_cnt (numeric, 22 distinct): ['3.0', '6.0', '1.0', '2.0', '5.0', '4.0', '7.0', '8.0', '9.0', '10.0']
score_1 (numeric, 1275 distinct): ['601.9791', '612.2197', '632.9522', '634.3876', '627.0548', '600.5437', '572.6966', '611.1168', '619.6559', '625.6194']
score_2 (numeric, 48 distinct): ['563.0898', '518.8954', '572.357', '576.8753', '546.4394', '564.0841', '554.8169', '535.5457', '529.2139', '556.7579']
age (numeric, 51 distinct): ['28', '27', '31', '30', '26', '32', '24', '29', '33', '35']
gender (numeric, 2 distinct): ['0', '1']
rep_loan_date_year (numeric, 3 distinct): ['2016', '2015', '2017']
rep_loan_date_month (numeric, 12 distinct): ['10', '11', '9', '8', '12', '5', '3', '4', '6', '7']
rep_loan_date_day (numeric, 31 distinct): ['27', '29', '16', '26', '18', '20', '10', '21', '12', '24']
rep_loan_date_weekday (numeric, 7 distinct): ['2', '3', '4', '0', '1', '6', '5']
first_loan_year (numeric, 2 distinct): ['2015', '2016']
first_loan_month (numeric, 12 distinct): ['3', '4', '10', '11', '9', '5', '8', '12', '2', '6']
first_loan_day (numeric, 31 distinct): ['13', '21', '15', '11', '16', '28', '7', '29', '30', '2']
first_loan_weekday (numeric, 7 distinct): ['2', '3', '6', '0', '4', '5', '1']
first_overdue_date_year (numeric, 3 distinct): ['2016.0', '2015.0']
first_overdue_date_month (numeric, 13 distinct): ['1.0', '5.0', '6.0', '4.0', '7.0', '11.0', '10.0', '8.0', '3.0', '12.0']
first_overdue_date_day (numeric, 20 distinct): ['30.0', '20.0', '13.0', '6.0', '15.0', '1.0', '22.0', '14.0', '9.0', '8.0']
first_overdue_date_weekday (numeric, 8 distinct): ['-1', '0', '2', '1', '3', '5', '4', '6']
'''


CONTEXT = "Fraud Detection Dataset"
TARGET = CuratedTarget(raw_name="bad_flag", new_name="Bad Flag", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': "Not Fraud", '1': "Fraud"})
COLS_TO_DROP = ["payment_type_5"]
FEATURES = [CuratedFeature(raw_name="dpd_5_cnt", new_name="Days Past Due - DPD 5 Count"),
            CuratedFeature(raw_name="dpd_15_cnt", new_name="Days Past Due - DPD 15 Count"),
            CuratedFeature(raw_name="dpd_30_cnt", new_name="Days Past Due - DPD 30 Count"),
            CuratedFeature(raw_name="federal_district_nm", new_name="Federal District Name"),
            CuratedFeature(raw_name="gender", new_name="Gender", value_mapping={"1": "Man", "0": "Woman"}),
            CuratedFeature(raw_name="rep_loan_date_year", new_name="Reporting Loan Date Year"),
            CuratedFeature(raw_name="rep_loan_date_month", new_name="Reporting Loan Date Month"),
            CuratedFeature(raw_name="rep_loan_date_day", new_name="Reporting Loan Date Day"),
            CuratedFeature(raw_name="rep_loan_date_weekday", new_name="Reporting Loan Date Weekday"),
            CuratedFeature(raw_name="first_overdue_date_day", new_name="First Overdue Date Day"),
            CuratedFeature(raw_name="first_overdue_date_weekday", new_name="First Overdue Date Weekday",
                           value_mapping={"0": "Sunday", "1": "Monday", "2": "Tuesday", "3": "Wednesday",
                                          "4": "Thursday", "5": "Friday",
                                          "6": "Saturday", "-1": "Unknown"})
            ]
