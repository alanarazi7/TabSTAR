from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: forest_fires
====
Examples: 517
====
URL: https://www.openml.org/search?type=data&id=44962
====
Description: **Data Description**

The aim of this dataset is to predict the burned area of forest fires, in the northeast region of Portugal, by using meteorological and other data.

The output 'area' was first transformed with a $ln(x+1)$ function. Then, several Data Mining methods were applied. After fitting the models, the outputs were  post-processed with the inverse of the $ln(x+1)$ transform. Four different input setups were used.

**Attribute Description**

1. *X* - x-axis spatial coordinate within the Montesinho park map: 1 to 9
2. *Y* - y-axis spatial coordinate within the Montesinho park map: 2 to 9
3. *month* - month of the year: 'jan' to 'dec'
4. *day* - day of the week: 'mon' to 'sun'
5. *FFMC* - FFMC index from the FWI system: 18.7 to 96.20
6. *DMC* - DMC index from the FWI system: 1.1 to 291.3
7. *DC* - DC index from the FWI system: 7.9 to 860.6
8. *ISI* - ISI index from the FWI system: 0.0 to 56.10
9. *temp* - temperature in Celsius degrees: 2.2 to 33.30
10. *RH* - relative humidity in %: 15.0 to 100
11. *wind* - wind speed in km/h: 0.40 to 9.40
12. *rain* - outside rain in mm/m2 : 0.0 to 6.4
13. *area* - the burned area of the forest (in ha): 0.00 to 1090.84 (this target variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform).
====
Target Variable: area (numeric, 251 distinct): ['0.0', '1.94', '0.52', '3.71', '0.68', '6.43', '2.14', '1.95', '2.18', '1.75']
====
Features:

X (numeric, 9 distinct): ['4', '6', '2', '8', '7', '3', '1', '5', '9']
Y (numeric, 7 distinct): ['4', '5', '6', '3', '2', '9', '8']
month (string, 12 distinct): ['aug', 'sep', 'mar', 'jul', 'feb', 'jun', 'oct', 'apr', 'dec', 'jan']
day (string, 7 distinct): ['sun', 'fri', 'sat', 'mon', 'tue', 'thu', 'wed']
FFMC (numeric, 106 distinct): ['92.1', '91.6', '91.0', '91.7', '92.4', '93.7', '92.5', '94.8', '90.1', '92.9']
DMC (numeric, 215 distinct): ['99.0', '129.5', '231.1', '142.4', '35.8', '126.5', '108.4', '108.3', '137.0', '152.6']
DC (numeric, 219 distinct): ['745.3', '692.6', '692.3', '715.1', '698.6', '601.4', '80.8', '647.1', '764.0', '706.4']
ISI (numeric, 119 distinct): ['9.6', '7.1', '6.3', '8.4', '7.0', '6.2', '9.2', '7.5', '9.0', '8.1']
temp (numeric, 192 distinct): ['17.4', '19.6', '15.4', '20.6', '20.4', '21.9', '19.1', '15.9', '16.8', '20.1']
RH (numeric, 75 distinct): ['27', '39', '35', '43', '42', '45', '34', '33', '40', '46']
wind (numeric, 21 distinct): ['3.1', '2.2', '4.0', '4.9', '2.7', '5.4', '4.5', '3.6', '1.8', '5.8']
rain (numeric, 7 distinct): ['0.0', '0.2', '0.8', '1.0', '6.4', '0.4', '1.4']
'''

CONTEXT = "Portugal Forest Fires Burned Area Prediction"
TARGET = CuratedTarget(raw_name="area", new_name="Burned Area of Forest", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="X", new_name="X-axis Spatial Coordinate"),
            CuratedFeature(raw_name="Y", new_name="Y-axis Spatial Coordinate"),
            CuratedFeature(raw_name="month", new_name="Month of the Year"),
            CuratedFeature(raw_name="day", new_name="Day of the Week"),
            CuratedFeature(raw_name="FFMC", new_name="FFMC Index"),
            CuratedFeature(raw_name="DMC", new_name="DMC Index"),
            CuratedFeature(raw_name="DC", new_name="DC Index"),
            CuratedFeature(raw_name="ISI", new_name="ISI Index"),
            CuratedFeature(raw_name="temp", new_name="Temperature in Celsius"),
            CuratedFeature(raw_name="RH", new_name="Relative Humidity"),
            CuratedFeature(raw_name="wind", new_name="Wind Speed in km/h"),
            CuratedFeature(raw_name="rain", new_name="Outside Rain in mm/m2")]