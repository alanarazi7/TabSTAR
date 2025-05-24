from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Credit-Risk-Dataset
====
Examples: 32581
====
URL: https://www.openml.org/search?type=data&id=43454
====
Description: Detailed data description of Credit Risk dataset:



Feature Name
Description




person_age
Age


person_income
Annual Income


personhomeownership
Home ownership


personemplength
Employment length (in years)


loan_intent
Loan intent


loan_grade
Loan grade


loan_amnt
Loan amount


loanintrate
Interest rate


loan_status
Loan status (0 is non default 1 is default)


loanpercentincome
Percent income


cbpersondefaultonfile
Historical default


cbpresoncredhistlength
Credit history length
====
Target Variable: loan_status (numeric, 2 distinct): ['0', '1']
====
Features:

person_age (numeric, 58 distinct): ['23', '22', '24', '25', '26', '27', '28', '29', '30', '21']
person_income (numeric, 4295 distinct): ['60000', '30000', '50000', '40000', '45000', '75000', '48000', '65000', '70000', '42000']
person_home_ownership (string, 4 distinct): ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
person_emp_length (numeric, 37 distinct): ['0.0', '2.0', '3.0', '5.0', '1.0', '4.0', '6.0', '7.0', '8.0', '9.0']
loan_intent (string, 6 distinct): ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
loan_grade (string, 7 distinct): ['A', 'B', 'C', 'D', 'E', 'F', 'G']
loan_amnt (numeric, 753 distinct): ['10000', '5000', '12000', '6000', '15000', '8000', '4000', '3000', '20000', '7000']
loan_int_rate (numeric, 349 distinct): ['7.51', '10.99', '7.49', '7.88', '5.42', '7.9', '11.49', '9.99', '13.49', '6.03']
loan_percent_income (numeric, 77 distinct): ['0.1', '0.13', '0.08', '0.07', '0.11', '0.09', '0.14', '0.12', '0.06', '0.17']
cb_person_default_on_file (string, 2 distinct): ['N', 'Y']
cb_person_cred_hist_length (numeric, 29 distinct): ['2', '3', '4', '8', '7', '9', '5', '6', '10', '14']
'''

CONTEXT = "Credit Risk Default Prediction"
TARGET = CuratedTarget(raw_name="loan_status", new_name="Credit Loan Status", task_type=SupervisedTask.BINARY,
                       label_mapping={"0": "Non-Default", "1": "Default"})
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name="loan_intent",
                   value_mapping={"DEBTCONSOLIDATION": "Debt Consolidation", "HOMEIMPROVEMENT": "Home Improvement"}),
]
