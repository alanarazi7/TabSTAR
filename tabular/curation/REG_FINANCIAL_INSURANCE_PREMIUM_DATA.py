from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Insurance-Premium-Data
====
Examples: 1338
====
URL: https://www.openml.org/search?type=data&id=43463
====
Description: This Dataset is something I found online when I wanted to practice regression models. It is an openly available online dataset at multiple places. Though I do not know the exact origin and collection methodology of the data, I would recommend this dataset to everybody who is just beginning their journey in Data science.
====
Features:

age (numeric, 47 distinct): ['18', '19', '50', '51', '47', '46', '45', '20', '48', '52']
sex (string, 2 distinct): ['male', 'female']
bmi (numeric, 548 distinct): ['32.3', '28.31', '30.495', '30.875', '31.35', '30.8', '34.1', '28.88', '33.33', '35.2']
children (numeric, 6 distinct): ['0', '1', '2', '3', '4', '5']
smoker (string, 2 distinct): ['no', 'yes']
region (string, 4 distinct): ['southeast', 'southwest', 'northwest', 'northeast']
charges (numeric, 1337 distinct): ['1639.5631', '16884.924', '29330.9832', '2221.5644', '19798.0546', '13063.883', '13555.0049', '44202.6536', '10422.9166', '7243.8136']
'''

CONTEXT = "Insurance Premium Data"
TARGET = CuratedTarget(raw_name="charges", new_name="Insurance Premium Charges", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
