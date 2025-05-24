from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: stock
====
Examples: 950
====
URL: https://www.openml.org/search?type=data&id=223
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

This is a dataset obtained from the StatLib repository. Here is the included description:

 The data provided are daily stock prices from January 1988 through October 1991, for ten aerospace companies.

 Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at
 http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html
 Original source: StatLib repository. 
 Characteristics: 950 cases, 10 continuous attributes
====
Target Variable: company10 (numeric, 203 distinct): ['44.125', '50.0', '38.375', '45.5', '45.25', '51.625', '55.625', '47.875', '47.625', '44.625']
====
Features:

company1 (numeric, 415 distinct): ['48.25', '46.125', '47.0', '27.781', '21.063', '46.0', '46.875', '45.875', '27.172', '47.125']
company2 (numeric, 271 distinct): ['53.0', '50.75', '50.875', '50.25', '52.75', '50.125', '53.625', '58.0', '52.25', '50.5']
company3 (numeric, 94 distinct): ['20.0', '15.75', '19.875', '20.125', '21.5', '20.625', '21.75', '20.5', '21.125', '19.75']
company4 (numeric, 186 distinct): ['43.875', '42.625', '43.75', '45.0', '43.0', '43.125', '43.375', '40.0', '41.625', '42.75']
company5 (numeric, 373 distinct): ['62.0', '64.25', '63.0', '62.75', '64.0', '54.5', '63.25', '63.875', '48.875', '64.625']
company6 (numeric, 153 distinct): ['29.0', '29.875', '30.0', '18.0', '17.75', '17.0', '26.25', '17.5', '26.125', '30.25']
company7 (numeric, 203 distinct): ['68.0', '67.375', '66.5', '64.75', '67.0', '68.75', '80.0', '66.375', '69.625', '66.75']
company8 (numeric, 99 distinct): ['22.0', '22.25', '21.625', '27.625', '21.875', '26.875', '22.125', '21.0', '21.5', '27.125']
company9 (numeric, 155 distinct): ['46.875', '45.125', '45.25', '44.75', '47.625', '47.875', '45.0', '44.25', '42.25', '42.75']
'''

CONTEXT = "Stock Prices of Aerospace Companies between 1988-1991"
TARGET = CuratedTarget(raw_name='company10', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []