from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: delays_zurich_transport
====
Examples: 5465575
====
URL: https://www.openml.org/search?type=data&id=40753
====
Description: Zurich public transport delay data 2016-10-30 03:30:00 CET - 2016-11-27 01:20:00 CET cleaned and prepared at Open Data Day 2017.
====
Target Variable: delay (numeric, 4082 distinct): ['10.0', '15.0', '12.0', '11.0', '13.0', '14.0', '9.0', '16.0', '18.0', '17.0']
====
Features:

vehicle_type (string, 3 distinct): ['Tram', 'Bus', 'Trolley']
line_number (string, 68 distinct): ['33', '9', '11', '7', '13', '31', '32', '80', '4', '2']
direction (numeric, 2 distinct): ['2', '1']
stop_id (numeric, 1530 distinct): ['10557.0', '10801.0', '4260.0', '11360.0', '10532.0', '10622.0', '11156.0', '11248.0', '11380.0', '11246.0']
weekday (nominal, 7 distinct): ['3', '5', '4', '6', '2', '7', '1']
time (string, 3526 distinct): ['2016-11-22 07:40:00', '2016-11-02 07:40:00', '2016-11-01 07:40:00', '2016-11-03 07:40:00', '2016-11-21 07:40:00', '2016-11-01 07:20:00', '2016-10-31 07:40:00', '2016-11-22 07:20:00', '2016-11-04 07:40:00', '2016-11-15 07:20:00']
temp (numeric, 143 distinct): ['8.7', '7.5', '8.8', '7.7', '7.3', '7.6', '6.5', '6.8', '7.0', '6.9']
windspeed_max (numeric, 131 distinct): ['3.0', '2.2', '2.6', '2.5', '3.4', '3.5', '2.4', '2.0', '3.2', '2.3']
windspeed_avg (numeric, 66 distinct): ['2.1', '2.3', '1.9', '1.7', '2.0', '1.3', '1.5', '2.4', '2.2', '2.5']
precipitation (numeric, 15 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.9', '0.8', '1.0']
dew_point (numeric, 120 distinct): ['5.5', '5.6', '7.8', '5.7', '7.3', '5.4', '7.7', '7.5', '7.4', '3.0']
humidity (numeric, 46 distinct): ['85', '87', '88', '89', '84', '86', '93', '92', '83', '94']
hour (numeric, 23 distinct): ['18', '15', '12', '17', '9', '6', '8', '11', '14', '16']
dayminute (numeric, 132 distinct): ['1039.0', '1080.0', '1059.0', '1000.0', '1020.0', '1010.0', '1049.0', '980.0', '1100.0', '1030.0']
'''


CONTEXT = "Zurich Public Transport Delay"
TARGET = CuratedTarget(raw_name="delay", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="time", feat_type=FeatureType.DATE)]
