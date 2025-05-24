from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: turkibintalib/saudi-arabia-used-cars-dataset/UsedCarsSA_Clean_EN.csv
====
Examples: 8035
====
URL: https://www.kaggle.com/turkibintalib/saudi-arabia-used-cars-dataset/UsedCarsSA_Clean_EN.csv
====
Description: 
Saudi Arabia Used Cars Dataset
Used Cars Prices and Spec Scraped From Syarah Website

About Dataset
Content
The dataset contains 8248 records of used cars collected from syarah.com. Each row represents a used car with a link to its webpage. Other information regarding each car is the brand name, model, manufacturing year, origin, the color of the car, options, capacity of the engine, type of fuel, transmission type, the mileage that the car covered, region price, and negotiable.
For contacting: comets.sda@gmail.com

====
Features:

Make (object, 59 distinct): ['Toyota', 'Hyundai', 'Ford', 'Chevrolet', 'Nissan', 'GMC', 'Kia', 'Lexus', 'Mercedes', 'Mazda']
Type (object, 381 distinct): ['Land Cruiser', 'Camry', 'Hilux', 'Accent', 'Yukon', 'Tahoe', 'Sonata', 'Taurus', 'Elantra', 'Corolla']
Year (int64, 52 distinct): ['2016', '2015', '2017', '2018', '2019', '2014', '2020', '2013', '2012', '2011']
Origin (object, 4 distinct): ['Saudi', 'Gulf Arabic', 'Other', 'Unknown']
Color (object, 15 distinct): ['White', 'Black', 'Silver', 'Grey', 'Another Color', 'Brown', 'Red', 'Golden', 'Blue', 'Navy']
Options (object, 3 distinct): ['Full', 'Standard', 'Semi Full']
Engine_Size (float64, 75 distinct): ['3.5', '2.0', '2.5', '1.6', '2.4', '5.3', '4.0', '2.7', '4.6', '1.5']
Fuel_Type (object, 3 distinct): ['Gas', 'Diesel', 'Hybrid']
Gear_Type (object, 2 distinct): ['Automatic', 'Manual']
Mileage (int64, 2175 distinct): ['300000', '200000', '300', '100000', '90000', '30000', '180000', '400000', '130000', '120000']
Region (object, 27 distinct): ['Riyadh', 'Dammam', 'Jeddah', 'Qassim', 'Al-Medina', 'Al-Ahsa', 'Aseer', 'Makkah', 'Taef', 'Tabouk']
Price (int64, 541 distinct): ['0', '45000', '35000', '55000', '30000', '50000', '65000', '25000', '40000', '75000']
Negotiable (bool, 2 distinct): ['0', '1']
'''

CONTEXT = "Saudi Arabia Used Cars Price from Syarah Website"
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []

DESCRIPTION = '''
Saudi Arabia Used Cars Dataset
Used Cars Prices and Spec Scraped From Syarah Website

About Dataset
Content
The dataset contains 8248 records of used cars collected from syarah.com. Each row represents a used car with a link to its webpage. Other information regarding each car is the brand name, model, manufacturing year, origin, the color of the car, options, capacity of the engine, type of fuel, transmission type, the mileage that the car covered, region price, and negotiable.
For contacting: comets.sda@gmail.com
'''