from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Click_prediction_small
====
Examples: 39948
====
URL: https://www.openml.org/search?type=data&id=42733
====
Description: This is the same data as version 5 (OpenML ID = 1220) with '_id' features coded as nominal factor variables.

This dataset is a subset of the KDDCup 2012 track 2 data created by Manu Joseph and Harsh Raj for the paper

Joseph, M., & Raj, H. (2022). GATE: Gated Additive Tree Ensemble for Tabular Classification and Regression. arXiv preprint arXiv:2207.08548v4.

We retrieved the data from Dropbox.

Note: please read the Kaggle dataset description carefully before using this dataset. This dataset mostly contains IDs that should be looked up in other files that are not on OpenML and were not used as part of the benchmark in the paper mentioned above.


====
Target Variable: click (nominal, 2 distinct): ['0', '1']
====
Features:

impression (numeric, 99 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '10.0', '9.0']
url_hash (numeric, 6941 distinct): ['1.4340390157469403e+19', '1.205787899908646e+19', '7.903914528320191e+18', '4.2981186814246446e+18', '1.4531867648059392e+19', '1.3756257544627677e+19', '5.851252814446936e+18', '1.5145480155589095e+19', '1.4756578758696272e+19', '2.6928596198512824e+18']
ad_id (nominal, 19228 distinct): ['9027213', '20192676', '21522776', '20908196', '3048011', '20644045', '21163923', '3065545', '20017078', '20030165']
advertiser_id (nominal, 6064 distinct): ['27961', '23808', '23777', '1325', '23778', '1268', '385', '23807', '28698', '24354']
depth (numeric, 3 distinct): ['2', '1', '3']
position (numeric, 3 distinct): ['1', '2', '3']
query_id (numeric, 30748 distinct): ['0.0', '1.0', '2.0', '4.0', '8.0', '5.0', '3.0', '6.0', '7.0', '15.0']
keyword_id (nominal, 19803 distinct): ['0', '1', '2', '8', '3', '6', '4', '10', '5', '9']
title_id (nominal, 25321 distinct): ['0', '4', '2', '1', '3', '5', '7', '8', '9', '6']
description_id (nominal, 22381 distinct): ['0', '1', '5', '2', '4', '3', '6', '9', '7', '8']
user_id (nominal, 30114 distinct): ['0', '2', '187', '154', '61', '56', '52', '125', '229', '124']
'''

CONTEXT = "Consumer Click Prediction"
TARGET = CuratedTarget(raw_name="click", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []