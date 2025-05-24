from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: boston
====
URL: https://www.openml.org/search?type=data&id=531
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.
Variables in order:
CRIM     per capita crime rate by town
ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS    proportion of non-retail business acres per town
CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX      nitric oxides concentration (parts per 10 million)
RM       average number of rooms per dwelling
AGE      proportion of owner-occupied units built prior to 1940
DIS      weighted distances to five Boston employment centres
RAD      index of accessibility to radial highways
TAX      full-value property-tax rate per $10,000
PTRATIO  pupil-teacher ratio by town
B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT    % lower status of the population
MEDV     Median value of owner-occupied homes in $1000's


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: last
====
Target Variable: MEDV (numeric, 229 distinct): ['50.0', '25.0', '22.0', '21.7', '23.1', '19.4', '20.6', '13.8', '21.4', '20.1']
====
Features:

CRIM (numeric, 504 distinct): ['0.015', '14.3337', '0.0347', '0.0311', '0.0305', '0.0254', '0.025', '0.013', '0.0615', '0.055']
ZN (numeric, 26 distinct): ['0.0', '20.0', '80.0', '22.0', '12.5', '25.0', '40.0', '45.0', '30.0', '90.0']
INDUS (numeric, 76 distinct): ['18.1', '19.58', '8.14', '6.2', '21.89', '3.97', '9.9', '8.56', '10.59', '5.86']
CHAS (nominal, 2 distinct): ['0', '1']
NOX (numeric, 81 distinct): ['0.538', '0.713', '0.437', '0.871', '0.624', '0.489', '0.693', '0.605', '0.74', '0.544']
RM (numeric, 446 distinct): ['5.713', '6.167', '6.127', '6.229', '6.405', '6.417', '6.782', '6.951', '6.63', '6.312']
AGE (numeric, 356 distinct): ['100.0', '95.4', '96.0', '98.2', '97.9', '98.8', '87.9', '95.6', '97.0', '21.4']
DIS (numeric, 412 distinct): ['3.4952', '5.7209', '5.2873', '6.8147', '5.4007', '6.3361', '3.9454', '6.498', '4.7211', '4.8122']
RAD (nominal, 9 distinct): ['24', '5', '4', '3', '6', '2', '8', '1', '7']
TAX (numeric, 66 distinct): ['666.0', '307.0', '403.0', '437.0', '304.0', '264.0', '398.0', '384.0', '277.0', '224.0']
PTRATIO (numeric, 46 distinct): ['20.2', '14.7', '21.0', '17.8', '19.2', '17.4', '18.6', '19.1', '18.4', '16.6']
B (numeric, 357 distinct): ['396.9', '393.74', '395.24', '376.14', '394.72', '395.63', '392.8', '395.56', '390.94', '393.68']
LSTAT (numeric, 455 distinct): ['7.79', '14.1', '6.36', '18.13', '8.05', '5.29', '13.44', '7.44', '18.06', '5.49']
'''

CONTEXT = "Boston House Prices"
TARGET = CuratedTarget(raw_name='MEDV', new_name="Boston House Price Median Value in Thousands of USD",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name='CRIM', new_name="Per Capita Crime Rate by Town"),
            CuratedFeature(raw_name='ZN', new_name="Proportion of Residential Land Zoned for Lots Over 25,000 sq.ft."),
            CuratedFeature(raw_name='INDUS', new_name="Proportion of Non-Retail Business Acres Per Town"),
            CuratedFeature(raw_name='CHAS', new_name="Charles River Boundary", value_mapping={'0': "No", '1': "Yes"}),
            CuratedFeature(raw_name='NOX', new_name="Nitric Oxides Concentration (parts per 10 million)"),
            CuratedFeature(raw_name='RM', new_name="Average Number of Rooms Per Dwelling"),
            CuratedFeature(raw_name='AGE', new_name="Proportion of Owner-Occupied Units Built Prior to 1940"),
            CuratedFeature(raw_name='DIS', new_name="Weighted Distances to Five Boston Employment Centres"),
            CuratedFeature(raw_name='RAD', new_name="Index of Accessibility to Radial Highways"),
            CuratedFeature(raw_name='TAX', new_name="Full-Value Property-Tax Rate Per $10,000"),
            CuratedFeature(raw_name='PTRATIO', new_name="Pupil-Teacher Ratio by Town"),
            CuratedFeature(raw_name='B', new_name="Blacks by town"),
            CuratedFeature(raw_name="LSTAT", new_name="Proportion of lower status population")
            ]
