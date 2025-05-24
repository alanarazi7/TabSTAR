from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Multiclass_Classification_for_Corporate_Credit_Ratings
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=46372
====
Description: This dataset is derived from the Credit Risk Analytics book by Harald, Daniel, and Bart, as described in the Medium article by Roi Polanitzer. It focuses on predicting financial difficulties and defaults in corporate credit ratings, which is crucial in the business world for stakeholders like banks and insurance companies.
====
Target Variable: rating (nominal, 10 distinct): ['above_average', 'average', 'bad', 'below_average', 'excellent', 'good', 'outstanding', 'poor', 'very_bad', 'very_good']
====
Features:

spid (numeric, 4969 distinct): ['312072', '245009', '157661', '345396', '213713', '213268', '374416', '359324', '299117', '169671']
commeqta (numeric, 4927 distinct): ['0.0884', '0.0705', '0.0856', '0.081', '0.0238', '0.0707', '0.0904', '0.1353', '0.0633', '0.0448']
llploans (numeric, 4698 distinct): ['0.0031', '0.0012', '0.0079', '0.0017', '0.0273', '0.0015', '0.0018', '0.0028', '0.0027', '0.0022']
costtoincome (numeric, 4987 distinct): ['0.4797', '0.4044', '0.6866', '0.4226', '0.5013', '0.3551', '0.6628', '0.5635', '0.6166', '0.6726']
roe (numeric, 4972 distinct): ['0.1077', '0.1013', '0.2621', '0.1362', '0.1016', '-0.0372', '0.2352', '0.1118', '0.1144', '0.0276']
liqassta (numeric, 4976 distinct): ['0.0438', '0.4446', '0.2711', '0.2476', '0.457', '0.0386', '0.138', '0.4098', '0.1298', '0.3844']
size (numeric, 4997 distinct): ['19.0723', '17.6652', '18.3629', '18.819', '15.9425', '16.4695', '15.6832', '19.1285', '21.1344', '16.2477']
'''

CONTEXT = "Corporate Credit Risk Rating"
TARGET = CuratedTarget(raw_name="rating", new_name="Company Credit", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="commeqta", new_name="Commercial Equity"),
            CuratedFeature(raw_name="llploans", new_name="LLP Loans"),
            CuratedFeature(raw_name="costtoincome", new_name="Cost to Income"),
            CuratedFeature(raw_name="roe", new_name="Return on Equity"),
            CuratedFeature(raw_name="liqassta", new_name="Liquid Assets"),
            ]
