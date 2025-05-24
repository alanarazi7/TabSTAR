from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita/mercedes-benz.csv
====
Examples: 16392
====
URL: https://www.kaggle.com/bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita/mercedes-benz.csv
====
Description: 
Second-hand Mercedes Benz price Italy
This is a dataset created by web scraping second-hand cars italian websites

About Dataset
The dataset include mercedes-benz cars for sale registered from 2000 to 2023
If you found this dataset usefull you can leave a like.

About columns:
brand:manufacturer
model:version of the car
first_reg:year of the first registration of the car
fuel: d for diesel, g for gas, e for electric, l for gpl
mileage_km: mileage of the car in km
seller_type:d for dealer, p for private
shift: manual or automatic
price:the price of the car
power_hp: the power expressed in horse power of the car

====
Features:

Unnamed: 0 (int64, 16392 distinct): ['0', '11269', '10919', '10920', '10921', '10922', '10923', '10924', '10925', '10926']
brand (object, 1 distinct): ['mercedes-benz']
model (object, 186 distinct): ['a 180', 'c 220', 'b 180', 'e 220', 'cla 200', 'glc 220', 'gla 200', 'glc 250', 'a 200', 'c 200']
first_reg (object, 160 distinct): ['12-2019', '07-2020', '03-2022', '03-2018', '10-2019', '03-2017', '03-2021', '02-2022', '11-2017', '10-2020']
fuel (object, 9 distinct): ['d', 'b', '2', '3', 'e', 'c', 'o', 'l', 'unknown']
mileage_km (object, 6282 distinct): ['150000', '130000', '120000', '160000', '100000', '170000', '115000', '90000', '125000', '140000']
seller_type (object, 2 distinct): ['d', 'p']
swift (object, 2 distinct): ['Automatic', 'Manual']
price (int64, 1737 distinct): ['18000', '29900', '18500', '19900', '26900', '19500', '25000', '27900', '28900', '24900']
power_hp (object, 226 distinct): ['136', '109', '170', '116', '194', '204', '150', '163', '258', '190']
'''

CONTEXT = "Second-hand cars Mercedes Benz price Italy"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['Unnamed: 0', "brand"]
FEATURES = [CuratedFeature(raw_name="first_reg", feat_type=FeatureType.DATE),]

DESCRIPTION = '''
Second-hand Mercedes Benz price Italy
This is a dataset created by web scraping second-hand cars italian websites

About Dataset
The dataset include mercedes-benz cars for sale registered from 2000 to 2023
If you found this dataset usefull you can leave a like.

About columns:
brand:manufacturer
model:version of the car
first_reg:year of the first registration of the car
fuel: d for diesel, g for gas, e for electric, l for gpl
mileage_km: mileage of the car in km
seller_type:d for dealer, p for private
shift: manual or automatic
price:the price of the car
power_hp: the power expressed in horse power of the car
'''