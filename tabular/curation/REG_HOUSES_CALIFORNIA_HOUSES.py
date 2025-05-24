from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: california_housing
====
URL: https://www.openml.org/search?type=data&id=44977
====
Description: **Data Description**

Information on the variables was collected using all the block groups in California from the 1990 Census. In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. Naturally, the geographical area included varies inversely with the population density. Distances among the centroids of each block group were computed as measured in latitude and longitude. All the block groups reporting zero entries for the independent and dependent variables were excluded. The final data contained 20,640 observations on 9 variables. 

Each row in the dataset represents one census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

The goal of the dataset is to predict the median house value. The original dataset description advised to predict the value using logarithmic transform.


**Attribute Description**

Census block group describing features:

1. *longitude* 
2. *latitude*
3. *housingMedianAge*
4. *totalRooms*
5. *totalBedrooms*
6. *population*
7. *households*
8. *medianIncome*
9. *medianHouseValue* - target feature
====
Target Variable: medianHouseValue (numeric, 3842 distinct): ['500001.0', '137500.0', '162500.0', '112500.0', '187500.0', '225000.0', '350000.0', '87500.0', '275000.0', '150000.0']
====
Features:

longitude (numeric, 844 distinct): ['-118.31', '-118.3', '-118.29', '-118.27', '-118.32', '-118.28', '-118.35', '-118.36', '-118.19', '-118.37']
latitude (numeric, 862 distinct): ['34.06', '34.05', '34.08', '34.07', '34.04', '34.09', '34.02', '34.1', '34.03', '33.93']
housingMedianAge (numeric, 52 distinct): ['52', '36', '35', '16', '17', '34', '26', '33', '18', '25']
totalRooms (numeric, 5926 distinct): ['1527.0', '1613.0', '1582.0', '2127.0', '1717.0', '2053.0', '1607.0', '1722.0', '1471.0', '1703.0']
totalBedrooms (numeric, 1928 distinct): ['280.0', '331.0', '343.0', '345.0', '394.0', '393.0', '328.0', '309.0', '348.0', '314.0']
population (numeric, 3888 distinct): ['891.0', '761.0', '1227.0', '1052.0', '850.0', '825.0', '782.0', '999.0', '1005.0', '753.0']
households (numeric, 1815 distinct): ['306.0', '386.0', '335.0', '282.0', '429.0', '375.0', '284.0', '297.0', '278.0', '340.0']
medianIncome (numeric, 12928 distinct): ['3.125', '15.0001', '2.875', '2.625', '4.125', '3.875', '3.375', '3.0', '4.0', '3.625']
'''

CONTEXT = "California Housing Prices"
TARGET = CuratedTarget(raw_name="medianHouseValue", new_name="Median House Value", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
