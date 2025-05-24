from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: climate_change_impact_on_agriculture_2024
====
Examples: 10000
====
URL: https://www.openml.org/search?type=data&id=46726
====
Description: Climate change has a profound impact on global agriculture, affecting crop yields, soil health, and farming sustainability. This synthetic dataset is designed to simulate real-world agricultural data, enabling researchers, data scientists, and policymakers to explore how climate variations influence food production across different regions.

*Key Features:
Climate Variables - Simulated data on temperature changes, precipitation levels, and extreme weather events
Crop Productivity - Modeled impact of climate shifts on yields of key crops like wheat, rice, and corn
Regional Insights - Includes various geographic regions to analyze diverse climate-agriculture interactions
Ideal for Predictive Modeling - Supports climate risk assessment, food security studies, and sustainability research

Dataset Overview:
This dataset has been synthetically generated and does not contain real-world agricultural records. It is intended for academic learning, climate impact analysis, and machine learning applications in environmental studies.

*Columns Description:
Region - Simulated geographic region
Year - Modeled year of data collection
Average_Temperature - Simulated temperature levels (C) in degrees Celsius
Precipitation - Modeled annual rainfall (mm)
Crop_Yield - Synthetic yield data for selected crops (tons/hectare)
Extreme_Weather_Events - Number of modeled extreme weather occurrences per year
Disclaimer:
This dataset is completely synthetic and should not be used for real-world climate policy decisions or agricultural forecasting. It is meant for educational purposes, research, and data science applications.

Use this dataset to analyze climate trends, build predictive models, and explore solutions for sustainable agriculture!

https://www.kaggle.com/datasets/waqi786/climate-change-impact-on-agriculture
====
Target Variable: Economic_Impact_Million_USD (numeric, 9631 distinct): ['616.6', '1012.69', '447.71', '227.76', '643.72', '270.4', '254.47', '323.32', '304.07', '396.57']
====
Features:

Year (numeric, 35 distinct): ['1999', '2019', '1991', '2012', '2004', '2013', '1994', '2001', '1996', '2023']
Country (string, 10 distinct): ['USA', 'Australia', 'China', 'Nigeria', 'India', 'Canada', 'Argentina', 'France', 'Russia', 'Brazil']
Region (string, 34 distinct): ['South', 'Northeast', 'North', 'Central', 'Punjab', 'Victoria', 'New South Wales', 'East', 'South West', 'Ontario']
Crop_Type (string, 10 distinct): ['Wheat', 'Cotton', 'Vegetables', 'Corn', 'Rice', 'Sugarcane', 'Fruits', 'Soybeans', 'Barley', 'Coffee']
Average_Temperature_C (numeric, 3677 distinct): ['3.41', '33.19', '20.35', '18.13', '0.15', '3.27', '25.43', '15.09', '11.75', '31.89']
Total_Precipitation_mm (numeric, 9784 distinct): ['1377.47', '2624.98', '2642.21', '2855.34', '679.97', '1505.05', '984.19', '1972.87', '2909.45', '865.03']
CO2_Emissions_MT (numeric, 2852 distinct): ['25.23', '5.34', '27.77', '17.71', '19.85', '22.76', '1.06', '1.14', '4.89', '11.81']
Crop_Yield_MT_per_HA (numeric, 850 distinct): ['1.53', '2.16', '1.8', '2.52', '0.9', '1.98', '2.97', '2.43', '2.7', '1.845']
Extreme_Weather_Events (numeric, 11 distinct): ['1', '6', '9', '3', '5', '10', '2', '0', '4', '7']
Irrigation_Access (numeric, 6003 distinct): ['79.29', '74.86', '19.24', '98.54', '80.44', '88.8', '95.29', '97.42', '52.43', '50.79']
Pesticide_Use_KG_per_HA (numeric, 4343 distinct): ['37.76', '44.69', '0.72', '2.78', '5.82', '3.61', '44.62', '20.73', '44.58', '17.24']
Fertilizer_Use_KG_per_HA (numeric, 6314 distinct): ['6.07', '77.99', '93.14', '65.33', '72.31', '93.93', '94.79', '86.47', '95.99', '57.22']
Soil_Health_Index (numeric, 5318 distinct): ['79.38', '63.03', '64.29', '63.62', '31.44', '98.05', '84.32', '98.97', '97.38', '61.44']
Adaptation_Strategies (string, 5 distinct): ['Water Management', 'No Adaptation', 'Drought-resistant Crops', 'Organic Farming', 'Crop Rotation']
'''

CONTEXT = "Climate Change impact on Agriculture"
TARGET = CuratedTarget(raw_name="Economic_Impact_Million_USD", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []