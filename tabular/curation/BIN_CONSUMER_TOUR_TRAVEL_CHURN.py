from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Tour-and-Travels-Customer-Churn-Prediction
====
Examples: 954
====
URL: https://www.openml.org/search?type=data&id=45545
====
Description: A Tour & Travels Company Wants To Predict Whether A Customer Will Churn Or Not Based On Indicators Given Below.

Help Build Predictive Models And Save The Company's Money.

Perform Fascinating EDAs.

The Data Was Used For Practice Purposes And Also During A Mini Hackathon, Its Completely Free To Use
====
Target Variable: Target (nominal, 2 distinct): ['0', '1']
====
Features:

Age (numeric, 11 distinct): ['30', '37', '34', '31', '28', '29', '36', '27', '35', '38']
FrequentFlyer (nominal, 3 distinct): ['No', 'Yes']
AnnualIncomeClass (nominal, 3 distinct): ['Middle Income', 'Low Income', 'High Income']
ServicesOpted (numeric, 6 distinct): ['1', '2', '3', '4', '5', '6']
AccountSyncedToSocialMedia (nominal, 2 distinct): ['No', 'Yes']
BookedHotelOrNot (nominal, 2 distinct): ['No', 'Yes']
'''

CONTEXT = "Tour and Travels Customer Churn Prediction"
TARGET = CuratedTarget(raw_name="Target", new_name="Churn", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
