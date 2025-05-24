from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Brazilian_houses
====
Examples: 10692
====
URL: https://www.openml.org/search?type=data&id=42688
====
Description: **Author**: Kaggle  
**Source**: [original](https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent) - 20-03-2020  
**Please cite**:   

This dataset contains 10962 houses to rent with 13 diferent features.

**Outliers **
Some values in the dataset can be considered as outliers for further analyses. Bear in mind that the Web Crawler was used only to get the data, so it's possible that errors in the original data exist.

**Changes in data between versions of dataset **
Since the WebCrawler was ran in different days for each version of dataset, there may be differences like added or deleted houses (as well as added cities).

Notes: 

1) This dataset corresponds to the 2nd version of the original dataset ("houses_to_rent_v2.csv").

2) The value '-' in the attribute floor was replaced by '0' as the data contributor stated that this refers to houses with just one floor (see https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent/discussion).
====
Target Variable: total_(BRL) (numeric, 5751 distinct): ['2555', '2633', '4089', '1219', '760', '1117', '1572', '2586', '10840', '2021']
====
Features:

city (nominal, 5 distinct): ['Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 'Campinas']
area (numeric, 517 distinct): ['50', '70', '60', '100', '80', '40', '90', '200', '45', '120']
rooms (numeric, 11 distinct): ['3', '2', '1', '4', '5', '6', '7', '8', '10', '13']
bathroom (numeric, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
parking_spaces (numeric, 11 distinct): ['1', '0', '2', '3', '4', '5', '6', '8', '7', '10']
floor (nominal, 35 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
animal (nominal, 2 distinct): ['acept', 'not acept']
furniture (nominal, 2 distinct): ['not furnished', 'furnished']
hoa_(BRL) (numeric, 1679 distinct): ['0', '400', '300', '500', '600', '450', '350', '700', '1000', '2000']
rent_amount_(BRL) (numeric, 1195 distinct): ['2500', '2000', '1200', '3000', '15000', '3500', '1800', '1500', '4000', '2200']
property_tax_(BRL) (numeric, 1243 distinct): ['0', '100', '50', '84', '250', '42', '167', '25', '67', '59']
fire_insurance_(BRL) (numeric, 216 distinct): ['16', '20', '26', '22', '14', '17', '23', '13', '18', '19']
'''

CONTEXT = "Prices of Brazilian Houses for rent"
TARGET = CuratedTarget(raw_name='total_(BRL)', new_name="Total Price in BRL", task_type=SupervisedTask.REGRESSION)
# The dataset is trivial if rent amount appears
COLS_TO_DROP = ['rent_amount_(BRL)']
FEATURES = []