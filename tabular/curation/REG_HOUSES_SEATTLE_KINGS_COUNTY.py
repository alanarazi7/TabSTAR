from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: house_sales
====
URL: https://www.openml.org/search?type=data&id=42731
====
Description: Date converted to year/mo/day numerics.This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

It contains 19 house features plus the price and the id columns, along with 21613 observations.
It's a great dataset for evaluating simple regression models.

Another version: https://www.openml.org/search?type=data&id=42079
====
Description: **Author**: https://www.kaggle.com/harlfoxem/  
https://www.kaggle.com/harlfoxem/  
**Source**: [original](https://www.kaggle.com/harlfoxem/housesalesprediction) - 2016-08-25  
**Please cite**:   

This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

It contains 19 house features plus the price and the id columns, along with 21613 observations.
It's a great dataset for evaluating simple regression models.

* Id: Unique ID for each home sold
* Date: Date of the home sale
* Price: Price of each home sold
* Bedrooms: Number of bedrooms
* Bathrooms: Number of bathrooms, where .5 accounts for a room with a toilet but no shower
* Sqft_living: Square footage of the apartments interior living space
* Sqft_lot: Square footage of the land space
* Floors: Number of floors
* Waterfront: A dummy variable for whether the apartment was overlooking the waterfront or not
* View: An index from 0 to 4 of how good the view of the property was
* Condition: An index from 1 to 5 on the condition of the apartment
* Grade: An index from 1 to 13, where 1-3 falls short of the building construction and design, 7 has an average level of  construction and design, and 11-13 have a high quality level of construction and design
* Sqft_above: The square footage of the interior housing space that is above ground level.
* Sqft_basement: The square footage of the interior housing space that is below ground level.
* Yr_built: The year the house was initially built
* Yr_renovated: The year of the house's last renovation
* Zipcode: What zipcode area the house is in
* Lat: Lattitude
* Long: Longitude
* Sqft_living15: The square footage of interior housing living space for the nearest 15 neighbors.
* Sqft_lot15: The square footage of the land lots of the nearest 15 neighbors.


====
Target Variable: price (numeric, 4028 distinct): ['350000.0', '450000.0', '550000.0', '500000.0', '425000.0', '325000.0', '400000.0', '375000.0', '300000.0', '525000.0']
====
Features:

bedrooms (numeric, 13 distinct): ['3', '4', '2', '5', '6', '1', '7', '0', '8', '9']
bathrooms (numeric, 30 distinct): ['2.5', '1.0', '1.75', '2.25', '2.0', '1.5', '2.75', '3.0', '3.5', '3.25']
sqft_living (numeric, 1038 distinct): ['1300', '1400', '1440', '1800', '1660', '1010', '1820', '1480', '1720', '1540']
sqft_lot (numeric, 9782 distinct): ['5000', '6000', '4000', '7200', '4800', '7500', '4500', '8400', '9600', '3600']
floors (numeric, 6 distinct): ['1.0', '2.0', '1.5', '3.0', '2.5', '3.5']
waterfront (numeric, 2 distinct): ['0', '1']
view (numeric, 5 distinct): ['0', '2', '3', '1', '4']
condition (numeric, 5 distinct): ['3', '4', '5', '2', '1']
grade (numeric, 12 distinct): ['7', '8', '9', '6', '10', '11', '5', '12', '4', '13']
sqft_above (numeric, 946 distinct): ['1300', '1010', '1200', '1220', '1140', '1400', '1060', '1180', '1340', '1250']
sqft_basement (numeric, 306 distinct): ['0', '600', '700', '500', '800', '400', '1000', '900', '300', '200']
yr_built (numeric, 116 distinct): ['2014', '2006', '2005', '2004', '2003', '2007', '1977', '1978', '1968', '2008']
yr_renovated (numeric, 70 distinct): ['0', '2014', '2013', '2003', '2005', '2007', '2000', '2004', '1990', '2006']
zipcode (nominal, 70 distinct): ['98103', '98038', '98115', '98052', '98117', '98042', '98034', '98118', '98023', '98006']
lat (numeric, 5034 distinct): ['47.6624', '47.5322', '47.6846', '47.5491', '47.6955', '47.6886', '47.6711', '47.5402', '47.6842', '47.6904']
long (numeric, 752 distinct): ['-122.29', '-122.3', '-122.362', '-122.291', '-122.363', '-122.372', '-122.288', '-122.357', '-122.284', '-122.365']
sqft_living15 (numeric, 777 distinct): ['1540', '1440', '1560', '1500', '1460', '1580', '1610', '1720', '1800', '1620']
sqft_lot15 (numeric, 8689 distinct): ['5000', '4000', '6000', '7200', '4800', '7500', '8400', '3600', '4500', '5100']
date_year (numeric, 2 distinct): ['2014', '2015']
date_month (numeric, 12 distinct): ['5', '4', '7', '6', '8', '10', '3', '9', '12', '11']
date_day (numeric, 31 distinct): ['23', '9', '5', '24', '20', '16', '17', '27', '22', '13']
'''

CONTEXT = "Seattle House Sales"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
