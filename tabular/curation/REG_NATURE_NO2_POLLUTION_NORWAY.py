from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: no2
====
Examples: 500
====
URL: https://www.openml.org/search?type=data&id=547
====
Description: **Author**: Magne Aldrin (magne.aldrin@nr.no)  
**Source**: [StatLib](http://lib.stat.cmu.edu/datasets/) - 2004  
**Please cite**:   

The data are a subsample of 500 observations from a data set that originate in a study where air pollution at a road is
related to traffic volume and meteorological variables, collected by the Norwegian Public Roads Administration. The response variable (column 1) consist of hourly values of the logarithm of the concentration of NO2 (particles), measured at Alnabru in Oslo, Norway, between October 2001 and August 2003. 

The predictor variables (columns 2 to 8) are the logarithm of the number of cars per hour, temperature $$2$$ meter above ground (degree C), wind speed (meters/second), the temperature difference between $$25$$ and $$2$$ meters above ground (degree C), wind direction (degrees between 0 and 360), hour of day and day number from October 1. 2001.
====
Target Variable: no2_concentration (numeric, 385 distinct): ['2.4248', '3.8133', '3.4404', '2.7344', '3.1946', '3.7955', '4.2613', '3.1739', '3.4843', '2.8736']
====
Features:

cars_per_hour (numeric, 464 distinct): ['4.8122', '6.5751', '4.4543', '7.9956', '5.1475', '5.6664', '6.3936', '7.8272', '5.9375', '7.5974']
temperature_at_2m (numeric, 223 distinct): ['1.1', '-4.1', '2.1', '-2.8', '-4.2', '1.3', '-1.6', '4.9', '6.4', '-5.6']
wind_speed (numeric, 78 distinct): ['1.3', '2.1', '2.9', '3.5', '4.2', '1.6', '2.8', '1.7', '1.5', '2.5']
temperature_diff_2m_25m (numeric, 61 distinct): ['-0.1', '0.0', '-0.2', '0.3', '0.1', '0.2', '0.6', '0.4', '-0.3', '0.5']
wind_direction (numeric, 373 distinct): ['77.0', '73.0', '82.0', '78.0', '79.0', '80.0', '76.0', '85.0', '220.0', '212.0']
hour_of_day (numeric, 24 distinct): ['17', '5', '15', '6', '2', '21', '14', '20', '16', '8']
day (numeric, 287 distinct): ['576', '417', '573', '128', '95', '174', '409', '445', '141', '529']
'''

CONTEXT = "Norway Roads NO2 Pollution"
TARGET = CuratedTarget(raw_name="no2_concentration", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []