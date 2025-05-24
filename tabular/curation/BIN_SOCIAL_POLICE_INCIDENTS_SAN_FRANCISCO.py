from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: sf-police-incidents
====
Examples: 2215023
====
URL: https://www.openml.org/search?type=data&id=42732
====
Description: Incident reports from the San Franciso Police Department between January 2003 and May 2018, provided by the City and County of San Francisco. The dataset was downloaded on 05.11.2018. from [https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry]. For a description of all variables, checkout the homepage of the data provider. The original data was published under ODC Public Domain Dedication and Licence (PDDL) [https://opendatacommons.org/licenses/pddl/1.0/]. As target, the binary variable 'ViolentCrime' was created. A 'ViolentCrime' was defined as 'Category' %in% c('ASSAULT', 'ROBBERY', 'SEX OFFENSES, FORCIBLE', 'KIDNAPPING') | 'Descript' %in% c('GRAND THEFT PURSESNATCH', 'ATTEMPTED GRAND THEFT PURSESNATCH'). Additional date and time features 'Hour', 'DayOfWeek', 'Month', and 'Year' were created. The original variables 'Category', 'Descript', 'Date', 'Time', 'Resolution', 'Location', and 'PdId' were removed from the dataset. One record which contained the only missing value in the variable 'PdDistrict' was removed from the dataset. Using this dataset for machine learning was inspired by Nina Zumel's blogpost [http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/]. Note that incidents consist of multiple rows in the dataset when the crime belongs to more than one 'Category', which is indicated by the ID variable 'IncidntNum' (ignored by default).
====
Target Variable: ViolentCrime (nominal, 2 distinct): ['No', 'Yes']
====
Features:

Hour (numeric, 24 distinct): ['18', '17', '12', '19', '16', '15', '20', '22', '0', '14']
DayOfWeek (nominal, 7 distinct): ['6', '4', '7', '5', '3', '2', '1']
Month (nominal, 12 distinct): ['1', '3', '10', '4', '5', '8', '9', '7', '2', '11']
Year (nominal, 16 distinct): ['2015', '2017', '2013', '2016', '2014', '2003', '2004', '2005', '2008', '2012']
PdDistrict (nominal, 10 distinct): ['SOUTHERN', 'MISSION', 'NORTHERN', 'CENTRAL', 'BAYVIEW', 'INGLESIDE', 'TENDERLOIN', 'TARAVAL', 'PARK', 'RICHMOND']
Address (nominal, 25147 distinct): ['800 Block of BRYANT ST', '800 Block of MARKET ST', '2000 Block of MISSION ST', '1000 Block of POTRERO AV', '900 Block of MARKET ST', '0 Block of TURK ST', '0 Block of 6TH ST', '16TH ST / MISSION ST', '300 Block of ELLIS ST', '1000 Block of MARKET ST']
X (numeric, 37380 distinct): ['-122.4034', '-122.4065', '-122.4197', '-122.4076', '-122.4197', '-122.4065', '-122.4662', '-122.4756', '-122.4099', '-122.4072']
Y (numeric, 37402 distinct): ['37.7754', '37.7565', '37.7642', '37.7842', '37.7651', '37.7851', '37.7725', '37.7285', '37.7834', '37.7866']
'''

CONTEXT = "San Francisco Police Incidents"
TARGET = CuratedTarget(raw_name="ViolentCrime", new_name="Violent Crime", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []