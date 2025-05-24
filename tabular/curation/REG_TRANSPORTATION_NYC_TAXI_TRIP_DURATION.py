from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: New-York-Taxi-Trip-enriched-by-Mathematica
====
Examples: 2083778
====
URL: https://www.openml.org/search?type=data&id=43584
====
Description: Context
This data set was created to help Kaggle users in the New Your City Taxi Trip Duration competition. New features were generated using Wolfram Mathematica system.
Hope that this data set will help both young and experienced researchers in their data mastering path.
All sources can be found here.
Content
Given dataset consists of both features from initial dataset and generated via Wolfram Mathematica computational system. Thus, all features can be split into following groups:

Initial features (extracted from initial data),
Calendar features (contains of season, day name and day period),
Weather features (information about temperature, snow, and rain),
Travel features (geo distance with estimated driving distance and time).

Dataset contains the following columns:

id - a unique identifier for each trip,
vendorId - a code indicating the provider associated with the trip record,
passengerCount - the number of passengers in the vehicle (driver entered value),
year,
month,
day,
hour,
minute,
second,
season,
dayName,
dayPeriod - day period, e.g. late night, morning, and etc.,
temperature,
rain,
snow,
startLatitude,
startLongitude,
endLatitude,
endLongitude,
flag - this flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip,
drivingDistance - driving distance, estimated via Wolfram Mathematica system,
drivingTime - driving time, estimated via Wolfram Mathematica system,
geoDistance - distance between starting and ending points,
tripDuration - duration of the trip in seconds (value -1 indicates test rows).
====
Features:

id (string, 2083778 distinct): ['id2875421', 'id2008303', 'id0882335', 'id3372095', 'id2788769', 'id0542364', 'id0858989', 'id0461193', 'id2701674', 'id3517298']
vendorId (numeric, 2 distinct): ['2', '1']
passengerCount (numeric, 10 distinct): ['1', '2', '5', '3', '6', '4', '0', '7', '9', '8']
year (numeric, 1 distinct): ['2016']
month (numeric, 6 distinct): ['3', '4', '5', '2', '6', '1']
day (numeric, 31 distinct): ['16', '14', '5', '12', '15', '9', '6', '4', '13', '19']
hour (numeric, 24 distinct): ['18', '19', '20', '21', '22', '17', '14', '15', '12', '13']
minute (numeric, 60 distinct): ['48', '49', '45', '50', '46', '54', '44', '47', '42', '52']
second (numeric, 60 distinct): ['3', '20', '30', '45', '11', '6', '35', '27', '22', '9']
season (string, 3 distinct): ['Spring', 'Winter', 'Summer']
dayName (string, 7 distinct): ['Friday', 'Saturday', 'Thursday', 'Wednesday', 'Tuesday', 'Sunday', 'Monday']
dayPeriod (string, 5 distinct): ['afternoon', 'morning', 'night', 'evening', 'lateNight']
temperature (numeric, 1800606 distinct): ['9.4', '3.3', '10.0', '-1.1', '15.6', '3.9', '-1.7', '1.7', '17.8', '7.8']
rain (numeric, 2 distinct): ['0', '1']
snow (numeric, 2 distinct): ['0', '1']
startLatitude (numeric, 48068 distinct): ['40.7741', '40.7741', '40.7741', '40.7741', '40.7741', '40.7741', '40.7741', '40.7742', '40.7741', '40.7741']
startLongitude (numeric, 24960 distinct): ['-73.9822', '-73.9821', '-73.9821', '-73.9821', '-73.9822', '-73.9822', '-73.9821', '-73.9822', '-73.9822', '-73.9822']
endLatitude (numeric, 67086 distinct): ['40.7743', '40.7743', '40.7501', '40.7502', '40.7743', '40.7501', '40.7501', '40.7501', '40.7743', '40.75']
endLongitude (numeric, 36977 distinct): ['-73.9823', '-73.9821', '-73.9822', '-73.9823', '-73.9914', '-73.9821', '-73.9822', '-73.9824', '-73.9824', '-73.9822']
flag (string, 2 distinct): ['N', 'Y']
drivingDistance (numeric, 1720328 distinct): ['0.0', '0.0004', '0.0008', '0.0006', '0.0002', '0.0002', '0.0004', '0.0006', '0.0006', '0.0008']
drivingTime (numeric, 231 distinct): ['135.0', '150.0', '165.0', '180.0', '195.0', '210.0', '120.0', '225.0', '240.0', '255.0']
geoDistance (numeric, 2075378 distinct): ['0.0', '20.3666', '2.0645', '819.9185', '1.8326', '47.3446', '560.2348', '0.4236', '0.4236', '942.752']
tripDuration (numeric, 7418 distinct): ['-1', '368', '408', '348', '367', '358', '399', '418', '417', '388']
'''

CONTEXT = "New York City Taxi Trip Duration"
TARGET = CuratedTarget(raw_name="tripDuration", new_name="Trip Duration", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id"]
FEATURES = []
