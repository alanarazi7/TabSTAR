from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Perth-House-Prices
====
URL: https://www.openml.org/search?type=data&id=43822
====
Description: Acknowledgements
This data was scraped from http://house.speakingsame.com/ and includes data from 322 Perth suburbs, resulting in an average of about 100 rows per suburb.
Content
I believe the columns chosen to represent this dataset are the most crucial in predicting house prices. Some preliminary analysis I conducted showed a significant correlation between each of these columns and the response variable (i.e. price). 
Data obtained from other than scrape source
Longitude and Latitude data was obtained from data.gov.au.
School ranking data was obtained from bettereducation.
The nearest schools to each address selected in this dataset are schools which are defined to be 'ATAR-applicable'. In the Australian secondary school education system, ATAR is a scoring system used to assess a student's culminative academic results and is used for entry into Australian universities. As such, schools which do not have an ATAR program such as primary schools, vocational schools, special needs schools etc. are not considered in determining the nearest school.
Do also note that under the "NEAREST_SCH_RANK" column, there are some missing rows as some schools are unranked according to this criteria by bettereducation.
====
Features:

SUBURB (string, 321 distinct): ['Bertram', 'Iluka', 'Bennett Springs', 'Mindarie', 'Carramar', 'Butler', 'Merriwa', 'Henley Brook', 'Darch', 'Jane Brook']
PRICE (numeric, 2297 distinct): ['430000', '400000', '450000', '500000', '420000', '480000', '550000', '460000', '520000', '510000']
BEDROOMS (numeric, 10 distinct): ['4', '3', '5', '2', '6', '1', '7', '8', '9', '10']
BATHROOMS (numeric, 8 distinct): ['2', '1', '3', '4', '5', '6', '7', '16']
GARAGE (numeric, 2503 distinct): ['2.0', '1.0', '3.0', '4.0', '6.0', '5.0', '8.0', '7.0', '12.0', '10.0']
LAND_AREA (numeric, 4372 distinct): ['700', '728', '1012', '450', '809', '600', '683', '680', '688', '510']
FLOOR_AREA (numeric, 528 distinct): ['200', '150', '160', '130', '180', '120', '140', '100', '170', '110']
BUILD_YEAR (numeric, 3279 distinct): ['2000.0', '2006.0', '2004.0', '2002.0', '2003.0', '2007.0', '2005.0', '1995.0', '2008.0', '2010.0']
CBD_DIST (numeric, 595 distinct): ['12600', '14900', '12500', '12400', '14800', '14700', '15800', '14600', '11900', '12200']
NEAREST_STN (string, 68 distinct): ['Midland Station', 'Warwick Station', 'Cockburn Central Station', 'Armadale Station', 'Butler Station', 'Currambine Station', 'Edgewater Station', 'Bull Creek Station', 'Murdoch Station', 'Warnbro Station']
NEAREST_STN_DIST (numeric, 1189 distinct): ['2100', '2000', '1900', '2200', '2300', '2400', '1800', '1700', '1500', '1100']
DATE_SOLD (string, 350 distinct): ['10-2020', '11-2020', '09-2020', '07-2019', '11-2018', '05-2018', '10-2018', '03-2018', '07-2020', '10-2019']
POSTCODE (numeric, 114 distinct): ['6056', '6065', '6164', '6112', '6167', '6055', '6030', '6027', '6163', '6107']
LATITUDE (numeric, 29707 distinct): ['-31.8848', '-32.0393', '-31.8254', '-32.1354', '-31.7877', '-31.802', '-31.7985', '-31.811', '-31.861', '-32.1252']
LONGITUDE (numeric, 28557 distinct): ['116.0004', '115.8098', '116.0187', '115.8207', '115.7446', '115.8316', '115.7678', '115.9735', '115.7533', '115.7574']
NEAREST_SCH (string, 160 distinct): ['SWAN VIEW SENIOR HIGH SCHOOL', 'KIARA COLLEGE', 'ATWELL COLLEGE', 'JOSEPH BANKS SECONDARY COLLEGE', 'SWAN VALLEY ANGLICAN COMMUNITY SCHOOL', 'MUNDARING CHRISTIAN COLLEGE', 'WANNEROO SECONDARY COLLEGE', 'YOUTH FUTURES COMMUNITY SCHOOL', 'COURT GRAMMAR SCHOOL', 'LAKE JOONDALUP BAPTIST COLLEGE']
NEAREST_SCH_DIST (numeric, 33318 distinct): ['0.7888', '0.8837', '0.969', '1.2158', '1.3129', '1.6232', '1.1921', '1.8691', '0.348', '0.2179']
NEAREST_SCH_RANK (numeric, 11055 distinct): ['53.0', '129.0', '92.0', '131.0', '35.0', '93.0', '102.0', '80.0', '61.0', '98.0']
'''

CONTEXT = "Perth House Prices"
TARGET = CuratedTarget(raw_name='PRICE', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
