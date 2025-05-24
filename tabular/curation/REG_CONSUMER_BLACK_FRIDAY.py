from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: black_friday
====
URL: https://www.openml.org/search?type=data&id=41540
====
Description: Customer purchases on Black Friday
====
Target Variable: Purchase (numeric, 13876 distinct): ['15879.0', '15756.0', '15206.0', '15891.0', '15616.0', '15916.0', '15909.0', '15349.0', '15544.0', '15339.0']
====
Features:

Gender (nominal, 2 distinct): ['M', 'F']
Age (nominal, 7 distinct): ['26-35', '36-45', '18-25', '46-50', '51-55', '55+', '0-17']
Occupation (numeric, 21 distinct): ['4', '0', '7', '17', '1', '12', '20', '14', '2', '16']
City_Category (nominal, 3 distinct): ['B', 'C', 'A']
Stay_In_Current_City_Years (nominal, 5 distinct): ['1', '2', '3', '4+', '0']
Marital_Status (numeric, 2 distinct): ['0', '1']
Product_Category_1 (numeric, 12 distinct): ['1', '5', '2', '3', '8', '6', '4', '11', '10', '13']
Product_Category_2 (numeric, 14 distinct): ['2', '8', '4', '5', '15', '6', '14', '13', '11', '9']
Product_Category_3 (numeric, 15 distinct): ['16', '15', '14', '17', '5', '8', '9', '12', '13', '6']
'''

CONTEXT = "Customer purchases on Black Friday"
TARGET = CuratedTarget(raw_name="Purchase", new_name="Purchase Total Amount", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
