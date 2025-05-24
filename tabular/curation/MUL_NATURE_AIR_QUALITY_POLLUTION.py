from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: air-quality-and-pollution-assessment
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=46762
====
Description: Environmental Metrics and Demographic Insights for Predicting Air Quality


About Dataset
This dataset focuses on air quality assessment across various regions. The dataset contains 5000 samples and captures critical environmental and demographic factors that influence pollution levels.

Key Features:

Temperature (C): Average temperature of the region.
Humidity (percentage): Relative humidity recorded in the region.
PM2.5 Concentration (ug_per_m3): Fine particulate matter levels.
PM10 Concentration (ug_per_m3): Coarse particulate matter levels.
NO2 Concentration (ppb): Nitrogen dioxide levels.
SO2 Concentration (ppb): Sulfur dioxide levels.
CO Concentration (ppm): Carbon monoxide levels.
Proximity to Industrial Areas (km): Distance to the nearest industrial zone.
Population Density (people/km2): Number of people per square kilometer in the region.
Target Variable: Air Quality Levels

Good: Clean air with low pollution levels.
Moderate: Acceptable air quality but with some pollutants present.
Poor: Noticeable pollution that may cause health issues for sensitive groups.
Hazardous: Highly polluted air posing serious health risks to the population.

This dataset is derived from several real-world sources that monitor air quality and environmental factors:

World Health Organization (WHO) (https://www.who.int/health-topics/air-pollution)

World Bank Data (https://data.worldbank.org/indicator/EN.POP.DNST)

https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment
====
Target Variable: Air_Quality (string, 4 distinct): ['Good', 'Moderate', 'Poor', 'Hazardous']
====
Features:

Temperature (numeric, 362 distinct): ['26.8', '26.7', '29.4', '26.3', '27.4', '23.6', '24.6', '32.2', '27.8', '26.2']
Humidity (numeric, 723 distinct): ['73.0', '72.5', '67.6', '64.4', '64.6', '60.1', '67.9', '72.7', '65.8', '75.6']
PM2.5 (numeric, 815 distinct): ['1.5', '1.1', '2.0', '0.7', '0.4', '2.3', '0.3', '2.8', '1.0', '2.5']
PM10 (numeric, 955 distinct): ['8.1', '16.3', '10.9', '14.1', '18.9', '8.4', '8.8', '8.0', '15.5', '14.9']
NO2 (numeric, 445 distinct): ['24.2', '25.3', '26.6', '23.1', '23.0', '23.5', '25.4', '20.9', '23.4', '22.9']
SO2 (numeric, 348 distinct): ['5.7', '5.9', '4.5', '4.9', '5.3', '6.3', '5.0', '6.4', '4.6', '5.1']
CO (numeric, 265 distinct): ['0.98', '0.99', '1.02', '1.03', '1.01', '1.04', '0.94', '0.97', '0.92', '1.0']
Proximity_to_Industrial_Areas (numeric, 179 distinct): ['5.1', '10.2', '10.3', '10.1', '5.2', '5.4', '10.4', '5.6', '10.5', '11.1']
Population_Density (numeric, 683 distinct): ['494', '454', '543', '471', '501', '538', '511', '438', '506', '485']
'''

CONTEXT = "Air Quality and Pollution Assessment from various regions"
TARGET = CuratedTarget(raw_name="Air_Quality", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []