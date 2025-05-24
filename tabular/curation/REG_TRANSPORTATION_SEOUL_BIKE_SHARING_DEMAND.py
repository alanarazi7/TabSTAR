from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: seoul_bike_sharing_demand_cat
====
Examples: 8760
====
URL: https://www.openml.org/search?type=data&id=46328
====
Description: From original source:
-----

The dataset contains count of public bicycles rented per hour in the Seoul Bike Sharing System, with corresponding weather data and holiday information

Additional Information

Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes. 
The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information. 

Has Missing Values?

No
-----
Columns with index [0] are dates and were dates and they were converted to colums ('day', 'month', 'year', 'week_day', 'timestamp').
====
Target Variable: rented_bike_count (numeric, 2166 distinct): ['0', '122', '223', '262', '165', '103', '189', '178', '170', '71']
====
Features:

Hour (numeric, 24 distinct): ['0', '1', '22', '21', '20', '19', '18', '17', '16', '15']
Temperature(C) (numeric, 546 distinct): ['19.1', '20.5', '23.4', '7.6', '20.7', '24.2', '20.2', '19.4', '19.0', '18.8']
Humidity(%) (numeric, 90 distinct): ['53', '97', '43', '57', '56', '47', '51', '63', '54', '52']
Wind speed (m/s) (numeric, 65 distinct): ['1.1', '1.2', '1.0', '0.9', '0.8', '1.4', '1.3', '1.5', '1.6', '0.6']
Visibility (10m) (numeric, 1789 distinct): ['2000', '1995', '1985', '1999', '1989', '1996', '1992', '1998', '1981', '1987']
Dew point temperature(C) (numeric, 556 distinct): ['0.0', '21.1', '14.3', '21.2', '8.9', '21.8', '2.2', '21.3', '20.2', '21.5']
Solar Radiation (MJ/m2) (numeric, 345 distinct): ['0.0', '0.01', '0.02', '0.03', '0.06', '0.05', '0.04', '0.11', '0.07', '0.16']
Rainfall(mm) (numeric, 61 distinct): ['0.0', '0.5', '1.0', '1.5', '0.1', '2.0', '2.5', '0.2', '3.5', '0.4']
Snowfall (cm) (numeric, 51 distinct): ['0.0', '0.3', '1.0', '0.9', '0.5', '0.7', '0.8', '2.0', '0.4', '1.6']
Seasons (nominal, 4 distinct): ['Spring', 'Summer', 'Autumn', 'Winter']
Holiday (nominal, 2 distinct): ['No Holiday', 'Holiday']
Functioning Day (nominal, 2 distinct): ['Yes', 'No']
day (numeric, 31 distinct): ['1', '2', '28', '27', '26', '25', '24', '23', '22', '21']
month (numeric, 12 distinct): ['12', '1', '3', '5', '7', '8', '10', '4', '6', '9']
year (numeric, 2 distinct): ['2018', '2017']
week_day (nominal, 7 distinct): ['4', '0', '1', '2', '3', '5', '6']
timestamp (numeric, 365 distinct): ['1512086400', '1533772800', '1533600000', '1533513600', '1533427200', '1533340800', '1533254400', '1533168000', '1533081600', '1532995200']
'''

CONTEXT = "Seoul Bike Sharing Demand"
TARGET = CuratedTarget(raw_name="rented_bike_count", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["day", "month", "year", "week_day"]
FEATURES = [CuratedFeature(raw_name="timestamp", feat_type=FeatureType.DATE)]
