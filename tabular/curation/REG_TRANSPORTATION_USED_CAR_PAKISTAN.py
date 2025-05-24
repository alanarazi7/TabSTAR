from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mustafaimam/used-car-prices-in-pakistan-2021/Used_car_prices_in_Pakistan_cleaned.csv
====
Examples: 72655
====
URL: https://www.kaggle.com/mustafaimam/used-car-prices-in-pakistan-2021/Used_car_prices_in_Pakistan_cleaned.csv
====
Description: 
Used car prices in Pakistan 2021
Dataset scraped from Pakwheels for the used cars for ML and Data science

About Dataset
This is the most recent data on the internet, of the used cars along with their prices to be sold in the local Pakistani market. This data is being scraped from the Pakisatns no. 1 used car selling website Pakwheels. In this data set, we have multiple features which play an important role in suggesting and evaluating the price of the specific car. In this dataset, we have car models from 1990 till 2021 including more than 280 unique model variants. As most of the available Pakistan used cars datasets were outdated and have incomplete information with mixed up labels and attributes, we created our own dataset using web scraping technique in python to extract data of used vehicles from www.pakwheels.com. Our dataset is updated with all appropriate and significant information required for prediction and we have cleaned null value as well as less-significant information provided by the user of the vehicle. The features available in this dataset are Make, Model, Version, Price, Make Year, CC, Assembly, Mileage, Registered City and Transmission. With 72656 examples.

====
Features:

Make (object, 52 distinct): ['Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'KIA', 'Hyundai', 'Nissan', 'Mitsubishi', 'Mercedes', 'FAW']
Model (object, 280 distinct): ['Corolla', 'Civic', 'Mehran', 'Cultus', 'City', 'Alto', 'Vitz', 'Wagon', 'Bolan', 'Vezel']
Version (object, 1328 distinct): ['GLi 1.3 VVTi', 'VXR', 'VX Euro II', 'VX', 'Oriel 1.8 i-VTEC CVT', 'VXR (CNG)', 'XLi VVTi', '1.3 i-VTEC', 'VXR Euro II', 'F 1.0']
Price (float64, 2156 distinct): ['750000.0', '650000.0', '1650000.0', '1250000.0', '850000.0', '1350000.0', '1450000.0', '1750000.0', '2250000.0', '2500000.0']
Make_Year (int64, 32 distinct): ['2017', '2018', '2015', '2016', '2014', '2019', '2007', '2012', '2013', '2021']
CC (int64, 80 distinct): ['1300', '1000', '800', '1500', '1800', '660', '1600', '2000', '2700', '3000']
Assembly (object, 2 distinct): ['Local', 'Imported']
Mileage (int64, 8132 distinct): ['100000', '150000', '80000', '200000', '70000', '90000', '1', '50000', '60000', '85000']
Registered City (object, 182 distinct): ['Lahore', 'Karachi', 'Islamabad', 'Un-Registered', 'Multan', 'Rawalpindi', 'Faisalabad', 'Peshawar', 'Sialkot', 'Gujranwala']
Transmission (object, 2 distinct): ['Manual', 'Automatic']
'''

CONTEXT = "Used car prices in Pakistan 2021"
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []

DESCRIPTION = '''
Used car prices in Pakistan 2021
Dataset scraped from Pakwheels for the used cars for ML and Data science

About Dataset
This is the most recent data on the internet, of the used cars along with their prices to be sold in the local Pakistani market. This data is being scraped from the Pakisatns no. 1 used car selling website Pakwheels. In this data set, we have multiple features which play an important role in suggesting and evaluating the price of the specific car. In this dataset, we have car models from 1990 till 2021 including more than 280 unique model variants. As most of the available Pakistan used cars datasets were outdated and have incomplete information with mixed up labels and attributes, we created our own dataset using web scraping technique in python to extract data of used vehicles from www.pakwheels.com. Our dataset is updated with all appropriate and significant information required for prediction and we have cleaned null value as well as less-significant information provided by the user of the vehicle. The features available in this dataset are Make, Model, Version, Price, Make Year, CC, Assembly, Mileage, Registered City and Transmission. With 72656 examples.
'''