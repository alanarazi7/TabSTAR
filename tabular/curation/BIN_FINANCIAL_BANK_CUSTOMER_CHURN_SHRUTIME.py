from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: shrutime
====
Examples: 10000
====
URL: https://www.openml.org/search?type=data&id=45062
====
Description: This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.Source: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
====
Target Variable: class (numeric, 2 distinct): ['0', '1']
====
Features:

CreditScore (numeric, 460 distinct): ['850', '678', '655', '667', '705', '684', '670', '651', '648', '652']
Age (numeric, 70 distinct): ['37', '38', '35', '36', '34', '33', '40', '39', '32', '31']
Balance (numeric, 6382 distinct): ['0.0', '105473.74', '130170.82', '159900.38', '117837.43', '144094.2', '124513.66', '109570.21', '111156.52', '121629.22']
EstimatedSalary (numeric, 9999 distinct): ['24924.92', '138552.74', '75888.65', '181610.6', '186275.7', '125877.22', '25329.48', '40066.95', '19458.75', '127154.8']
Geography (nominal, 3 distinct): ['France', 'Germany', 'Spain']
IsActiveMember (nominal, 2 distinct): ['1', '0']
Tenure (nominal, 11 distinct): ['2', '1', '7', '8', '5', '3', '4', '9', '6', '10']
Gender (nominal, 2 distinct): ['Male', 'Female']
HasCrCard (nominal, 2 distinct): ['1', '0']
NumOfProducts (nominal, 4 distinct): ['1', '2', '3', '4']
'''

CONTEXT = "Bank Customers Churn Prediction"
TARGET = CuratedTarget(raw_name="class", new_name="Churned", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="HasCrCard", new_name="Has Credit Card", value_mapping={'1': 'Yes', '0': 'No'})]
