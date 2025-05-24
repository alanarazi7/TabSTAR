from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: New-York-Citi-Bike-Trip-Duration-2016
====
Examples: 4500000
====
URL: https://www.openml.org/search?type=data&id=43573
====
Description: Context
Inspired by the New York City Taxi Trip Duration playground I created a dataset using the publicly available data from this link). Citi Bike is a bike sharing service available in New York City, that permits easy and affordable bike trips. They regularly release data about such trips, including starting and ending stations, starting and ending time, duration of the trip and few others variables.
It closely resembles the data available about taxi trips and I think it could be interesting to compare the two datasets. Let me know if you have any comment.
Content
The dataset covers 4.5M Citi Bike trips from the first 6 months of 2016. The data has been anonymized and the content has been arranged to follow the Taxi Trip dataset categories and nomenclature. 
Notice that the starting and ending point of each trip correspond to one of the 500 Citi Bike stations spread around NYC, most of them in Manhattan, with a substantial subset in Brooklyn.
Acknowledgements
This dataset is the property of NYC Bike Share, LLC and Jersey City Bike Share, LLC (Bikeshare) operates New York Citys Citi Bike bicycle sharing service for TC click here
Inspiration
Is there a correlation between the duration of bike rides and taxi rides? Weather or traffic conditions could affect both in a similar way. 
Is it always faster to get a cab?
====
Features:

gender_id (numeric, 3 distinct): ['1', '2', '0']
pickup_datetime (string, 3465183 distinct): ['2016-06-04 13:39:41', '2016-06-14 17:39:10', '2016-06-04 14:08:12', '2016-04-19 08:48:45', '2016-06-22 17:17:58', '2016-05-17 08:40:43', '2016-04-18 17:44:34', '2016-04-18 18:17:18', '2016-06-27 18:09:29', '2016-06-21 18:30:22']
dropoff_datetime (string, 3465441 distinct): ['2016-06-10 18:52:35', '2016-06-04 16:27:10', '2016-03-30 17:45:25', '2016-06-21 18:18:32', '2016-06-20 18:35:40', '2016-05-31 18:00:35', '2016-04-22 09:36:54', '2016-06-01 18:33:45', '2016-05-12 08:58:50', '2016-06-21 18:35:00']
pickup_longitude (numeric, 492 distinct): ['-73.9777', '-73.9942', '-73.9908', '-73.9901', '-73.9896', '-74.0132', '-74.0078', '-73.9939', '-73.9907', '-73.9972']
pickup_latitude (numeric, 492 distinct): ['40.7519', '40.7417', '40.7303', '40.737', '40.7403', '40.7175', '40.7467', '40.7516', '40.7345', '40.7221']
dropoff_longitude (numeric, 511 distinct): ['-73.9777', '-73.9942', '-73.9908', '-73.9901', '-74.0132', '-73.9896', '-74.0078', '-73.99', '-74.0026', '-73.9907']
dropoff_latitude (numeric, 511 distinct): ['40.7519', '40.7417', '40.7303', '40.737', '40.7175', '40.7403', '40.7467', '40.7564', '40.739', '40.7345']
trip_duration (numeric, 22288 distinct): ['345', '338', '370', '324', '366', '379', '342', '350', '363', '353']
'''

CONTEXT = "New York City Bike Trip Duration"
TARGET = CuratedTarget(raw_name="trip_duration", new_name="Trip Duration", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="pickup_datetime", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="dropoff_datetime", feat_type=FeatureType.DATE)]
