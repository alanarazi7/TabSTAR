from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Lisbon-House-Prices
====
URL: https://www.openml.org/search?type=data&id=43660
====
Description: Context
Explore the regression algorithm using the prices of Lisbon's houses. This dataset contains a total of 246 records. 
Content
The attributes of this dataset are:

Id: is a unique identifying number assigned to each house.
Condition: The house condition  (i.e., New, Used, As New, For Refurbishment).
PropertyType: Property type (i.e., Home, Single habitation) 
PropertySubType: Property Sub Type (i.e., Apartment, duplex, etc.)  
Bedrooms: Number of Bedrooms
Bathrooms: Number of Bathrooms
AreaNet: Net area of the house
AreaGross: Gross area of the house
Parking: Number of parking places
Latitude: Geographical Latitude
Longitude: Geographical Longitude
Country: Country where the house is located
District: District where the house is located
Municipality: Municipality where the house is located
Parish: Parish where the house is located
Price Sq. M.: Price per m in the location of the house
Price: This is our training variable and target. It is the home price.
====
Target Variable: Price (numeric, 160 distinct): ['295000', '375000', '700000', '450000', '500000', '315000', '275000', '350000', '490000', '515000']
====
Features:

Id (numeric, 246 distinct): ['101', '259', '261', '262', '263', '264', '265', '266', '267', '268']
Condition (string, 4 distinct): ['New', 'As New', 'Used', 'For Refurbishment']
PropertyType (string, 2 distinct): ['Homes', 'Single Habitation']
PropertySubType (string, 8 distinct): ['Apartment', 'Duplex', 'Townhouse Dwelling', 'Dwelling', 'Studio', 'Isolated Villa', 'Penthouse', 'Apart Hotel']
Bedrooms (numeric, 10 distinct): ['2', '3', '1', '4', '5', '0', '6', '7', '8', '11']
Bathrooms (numeric, 7 distinct): ['1', '2', '3', '4', '5', '6', '0']
AreaNet (numeric, 123 distinct): ['79', '50', '76', '130', '139', '145', '100', '60', '48', '90']
AreaGross (numeric, 123 distinct): ['158', '100', '152', '260', '278', '290', '200', '120', '96', '180']
Parking (numeric, 4 distinct): ['0', '2', '1', '3']
Latitude (numeric, 139 distinct): ['38.7458', '38.7214', '38.7116', '38.7137', '38.6963', '38.7176', '38.7118', '38.7177', '38.7269', '38.7169']
Longitude (numeric, 141 distinct): ['-9.0978', '-9.1607', '-9.1472', '-9.128', '-9.2202', '-9.1647', '-9.1649', '-9.1497', '-9.1286', '-9.1344']
Country (string, 1 distinct): ['Portugal']
District (string, 1 distinct): ['Lisboa']
Municipality (string, 1 distinct): ['Lisboa']
Parish (string, 24 distinct): ['Marvila', 'Campo de Ourique', 'Estrela', 'Santa Maria Maior', 'Arroios', 'Sao Vicente', 'Penha de Franca', 'Olivais', 'Belem', 'Santo Antonio']
Price_M2 (numeric, 24 distinct): ['2881', '3859', '4005', '4807', '3277', '3402', '2973', '2463', '3542', '5340']
'''

CONTEXT = "Lisbon House Prices"
TARGET = CuratedTarget(raw_name='Price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
