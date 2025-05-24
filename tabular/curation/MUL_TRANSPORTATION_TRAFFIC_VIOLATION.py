from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Traffic_violations
====
Examples: 70340
====
URL: https://www.openml.org/search?type=data&id=42345
====
Description: This dataset contains traffic violation information from all electronic traffic violations issued in the County. Any information that can be used to uniquely identify the vehicle, the vehicle owner or the officer issuing the violation will not be published. For this version, some features were removed and all remaining character features were recoded as nominal factor variables. All punctuation characters were removed from factor levels.
                      The variable 'Violation.Type' is used as target by default. The smaller target categories 'SERO' and 'ESERO' were collapsed into one category labeled 'SERO'. For this version, the dataset was downsampled to 5% of the original size. Unused factor levels and a few almost constant features were dropped.
====
Target Variable: Violation.Type (nominal, 3 distinct): ['Warning', 'Citation', 'SERO']
====
Features:

Description (nominal, 2130 distinct): ['DRIVER FAILURE TO OBEY PROPERLY PLACED TRAFFIC CONTROL DEVICE INSTRUCTIONS', 'FAILURE TO DISPLAY REGISTRATION CARD UPON DEMAND BY POLICE OFFICER', 'DRIVING VEHICLE ON HIGHWAY WITH SUSPENDED REGISTRATION', 'FAILURE OF INDIVIDUAL DRIVING ON HIGHWAY TO DISPLAY LICENSE TO UNIFORMED POLICE ON DEMAND', 'DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION', 'DISPLAYING EXPIRED REGISTRATION PLATE ISSUED BY ANY STATE', 'DRIVER FAILURE TO STOP AT STOP SIGN LINE', 'PERSON DRIVING MOTOR VEHICLE ON HIGHWAY OR PUBLIC USE PROPERTY ON SUSPENDED LICENSE AND PRIVILEGE', 'EXCEEDING THE POSTED SPEED LIMIT OF 30 MPH', 'DRIVING VEHICLE ON HIGHWAY WITHOUT CURRENT REGISTRATION PLATES AND VALIDATION TABS']
Belts (nominal, 2 distinct): ['No', 'Yes']
Personal.Injury (nominal, 2 distinct): ['No', 'Yes']
Property.Damage (nominal, 2 distinct): ['No', 'Yes']
Commercial.License (nominal, 2 distinct): ['No', 'Yes']
Commercial.Vehicle (nominal, 2 distinct): ['No', 'Yes']
State (nominal, 58 distinct): ['MD', 'VA', 'DC', 'XX', 'PA', 'FL', 'TX', 'NC', 'WV', 'NY']
VehicleType (nominal, 22 distinct): ['02 - Automobile', '05 - Light Duty Truck', '28 - Other', '03 - Station Wagon', '01 - Motorcycle', '06 - Heavy Duty Truck', '29 - Unknown', '08 - Recreational Vehicle', '25 - Utility Trailer', '07 - Truck/Road Tractor']
Year (numeric, 97 distinct): ['2006.0', '2005.0', '2007.0', '2004.0', '2003.0', '2008.0', '2012.0', '2013.0', '2002.0', '2011.0']
Make (nominal, 889 distinct): ['TOYOTA', 'HONDA', 'FORD', 'TOYT', 'NISSAN', 'HOND', 'CHEV', 'BMW', 'DODGE', 'ACURA']
Model (nominal, 3831 distinct): ['4S', 'TK', 'ACCORD', 'CIVIC', 'CAMRY', 'COROLLA', 'ALTIMA', '2S', '4D', 'SUV']
Color (nominal, 27 distinct): ['BLACK', 'SILVER', 'WHITE', 'GRAY', 'RED', 'BLUE', 'GREEN', 'GOLD', 'BLUE DARK', 'TAN']
Charge (nominal, 605 distinct): ['218011', '21201a1', '13409b', '13401h', '21707a', '16112c', '13411f', '2111242d2', '16101a', '16303c']
Contributed.To.Accident (nominal, 2 distinct): ['No', 'Yes']
Race (nominal, 6 distinct): ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER', 'NATIVE AMERICAN']
Gender (nominal, 3 distinct): ['M', 'F', 'U']
Driver.City (nominal, 1890 distinct): ['SILVER SPRING', 'GAITHERSBURG', 'GERMANTOWN', 'ROCKVILLE', 'WASHINGTON', 'BETHESDA', 'MONTGOMERY VILLAGE', 'HYATTSVILLE', 'POTOMAC', 'OLNEY']
Driver.State (nominal, 57 distinct): ['MD', 'DC', 'VA', 'PA', 'FL', 'NY', 'NC', 'WV', 'NJ', 'CA']
DL.State (nominal, 64 distinct): ['MD', 'VA', 'DC', 'XX', 'PA', 'FL', 'NY', 'CA', 'NC', 'NJ']
Arrest.Type (nominal, 19 distinct): ['A - Marked Patrol', 'Q - Marked Laser', 'B - Unmarked Patrol', 'S - License Plate Recognition', 'O - Foot Patrol', 'L - Motorcycle', 'E - Marked Stationary Radar', 'G - Marked Moving Radar (Stationary)', 'R - Unmarked Laser', 'I - Marked Moving Radar (Moving)']
'''

CONTEXT = "Traffic Violations"
TARGET = CuratedTarget(raw_name="Violation.Type", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"SERO": "SERO or ESERO"})
COLS_TO_DROP = []
FEATURES = []