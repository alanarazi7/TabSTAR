from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Insurance
====
Examples: 23548
====
URL: https://www.openml.org/search?type=data&id=45064
====
Description: This dataset is for classification tasks, and has both continuous and categorical variables.
====
Target Variable: class (numeric, 2 distinct): ['0', '1']
====
Features:

Upper_Age (numeric, 55 distinct): ['75', '28', '25', '27', '26', '52', '54', '55', '29', '48']
Lower_Age (numeric, 60 distinct): ['75', '28', '25', '26', '27', '24', '29', '32', '30', '33']
Reco_Policy_Premium (numeric, 5417 distinct): ['15288.0', '15600.0', '10224.0', '17640.0', '16128.0', '17136.0', '11880.0', '11700.0', '12760.0', '10944.0']
City_Code (nominal, 36 distinct): ['C1', 'C2', 'C3', 'C4', 'C9', 'C7', 'C8', 'C6', 'C10', 'C5']
Accomodation_Type (nominal, 2 distinct): ['Owned', 'Rented']
Reco_Insurance_Type (nominal, 2 distinct): ['Individual', 'Joint']
Is_Spouse (nominal, 2 distinct): ['No', 'Yes']
Health Indicator (nominal, 9 distinct): ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
Holding_Policy_Duration (nominal, 15 distinct): ['1.0', '2.0', '14+', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
Holding_Policy_Type (nominal, 4 distinct): ['3.0', '1.0', '2.0', '4.0']
'''

CONTEXT = "Health Insurance Lead Prediction"
TARGET = CuratedTarget(raw_name="class", new_name="Insurance", task_type=SupervisedTask.BINARY,
                       label_mapping={"0": "No", "1": "Yes"})
COLS_TO_DROP = []
FEATURES = []