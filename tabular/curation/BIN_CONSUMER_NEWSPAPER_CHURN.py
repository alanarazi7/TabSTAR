from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: NewspaperChurn
====
Examples: 15855
====
URL: https://www.openml.org/search?type=data&id=44226
====
Description: This dataset includes 15856 records of individuals who are or were subscribers to this newspaper. Datasets contain demographic information like HH Income which stands for the household income, home ownership, dummy for Children, Ethnicity, Year Of Residence, Age range, Language, and Nielsen Prizm; Geographic information like Address, State, City, County, and Zip Code. Also, the delivery period is chosen by the particular subscriber, as well as the weekly charge associated with it. The dataset also included the number of rewards subscribers used, and the source channel he/she been recruited in the first place. Finally, dataset included the information of whether the customer is still our subscriber or not.
====
Target Variable: Subscriber (string, 2 distinct): ['NO', 'YES']
====
Features:

SubscriptionID (numeric, 15855 distinct): ['180590686', '140203906', '180574686', '170132341', '150571344', '181453462', '180483689', '180540040', '181509544', '180486493']
HH Income (string, 18 distinct): ['Under $20,000', '$100,000 - $124,999', '$125,000 - $149,999', '$  20,000 - $29,999', '$  40,000 - $49,999', '$  30,000 - $39,999', '$  70,000 - $79,999', '$  50,000 - $59,999', '$150,000 - $174,999', '$  80,000 - $89,999']
Home Ownership (string, 2 distinct): ['OWNER', 'RENTER']
Ethnicity (string, 73 distinct): ['Hispanic', 'English', 'unknown', 'German', 'Scottish (Scotch)', 'Irish', 'Vietnamese', 'Italian', 'Chinese', 'Jewish']
dummy for Children (string, 2 distinct): ['N', 'Y']
Year Of Residence (numeric, 56 distinct): ['1', '2', '4', '3', '7', '6', '5', '23', '8', '10']
Age range (string, 13 distinct): ['50-54', '45-49', '55-59', '75 years or more', '60-64', '40-44', '65-69', '35-39', '30-34', '70-74']
Language (string, 38 distinct): ['English', 'Spanish', 'Vietnamese', 'Chinese', 'Korean', 'Portuguese', 'Hindi', 'Arabic', 'Italian', 'Japanese']
Address (string, 15742 distinct): ['1021 W BISHOP ST APT A', '2000 MAIN ST', '11152 TARAWA DR', '23212 ORANGE AVE APT 7', '1071 WALNUT AVE APT 22', '1102 E BUFFALO AVE', '1119 S GOLDEN WEST AVE', '1882 W HARRIET LN', '147 CORNELL', '3064 N SKYWOOD ST']
State (string, 1 distinct): ['CA']
City (string, 56 distinct): ['ANAHEIM', 'SANTA ANA', 'HUNTINGTON BEACH', 'IRVINE', 'ORANGE', 'GARDEN GROVE', 'FULLERTON', 'COSTA MESA', 'MISSION VIEJO', 'LONG BEACH']
County (string, 4 distinct): ['ORANGE', 'LOS ANGELES', 'RIVERSIDE', 'SAN BERNARDINO']
Zip Code (numeric, 117 distinct): ['92704', '92683', '92804', '92805', '92646', '92708', '92630', '90631', '92870', '92647']
weekly fee (string, 15 distinct): ['$0.01 - $0.50', '$0 - $0.01', '$1.00 - $1.99', '$2.00 - $2.99', '$4.00 - $4.99', '$3.00 - $3.99', '$0.51 - $0.99', '$10.00 - $10.99', '$5.00 - $5.99', '$7.00 - $7.99']
Deliveryperiod (string, 28 distinct): ['SunOnly', '7Day', 'Thu-Sun', 'SoooTFS', 'SatSun', '7DayOL', 'Soooooo', 'Fri-Sun', 'THU-SUN', '7DayT']
Nielsen Prizm (string, 10 distinct): ['MW', 'FM', 'MM', 'YM', 'FE', 'YW', 'FW', 'ME', 'YE']
reward program (numeric, 116 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7', '11']
Source Channel (string, 51 distinct): ['Partner', 'CustCall', 'Crew', 'TeleIn', 'DirectMl', 'Internet', 'CircAdm', 'Kiosk', 'TeleOut', 'System']
'''

CONTEXT = "Newspaper Customers Churn Prediction"
TARGET = CuratedTarget(raw_name="Subscriber", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["SubscriptionID"]
FEATURES = []
