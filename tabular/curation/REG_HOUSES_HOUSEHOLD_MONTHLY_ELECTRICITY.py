from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Household-monthly-electricity-bill
====
URL: https://www.openml.org/search?type=data&id=43588
====
Description: Introduction
The idea behind this dataset is to see how the number of people and the home size affects the monthly electricity consumption in the household.
Column decription:



Column
Explanation




num_rooms
Number of room in the house


num_people
Number of people in the house


housearea
Area of the house


is_ac
Is AC present in the house?


is_tv
Is TV present in the house?


is_flat
Is house a flat?


avemonthlyincome
Average monthly income of the household


num_children
Number of children in the house


is_urban
Is the house present in an urban area


amount_paid
Amount paid as the monthly bill



Acknowledgements
This dataset was prepared as a mock up dataset for practice use
====
Features:

num_rooms (numeric, 7 distinct): ['2', '1', '3', '0', '4', '-1', '5']
num_people (numeric, 13 distinct): ['5', '4', '6', '7', '3', '2', '8', '1', '9', '0']
housearea (numeric, 990 distinct): ['704.72', '761.19', '843.85', '924.05', '631.54', '873.78', '889.54', '825.89', '711.94', '635.28']
is_ac (numeric, 2 distinct): ['0', '1']
is_tv (numeric, 2 distinct): ['1', '0']
is_flat (numeric, 2 distinct): ['0', '1']
ave_monthly_income (numeric, 1000 distinct): ['9675.93', '13845.18', '30987.64', '10054.41', '10853.42', '16025.37', '14693.76', '29637.91', '5037.37', '16240.07']
num_children (numeric, 5 distinct): ['1', '0', '2', '3', '4']
is_urban (numeric, 2 distinct): ['1', '0']
amount_paid (numeric, 1000 distinct): ['560.4814', '717.6134', '510.6912', '366.9861', '768.9256', '405.2861', '695.5033', '720.9022', '606.0079', '779.35']
'''

CONTEXT = "Household Monthly Electricity Bill"
TARGET = CuratedTarget(raw_name='amount_paid', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
