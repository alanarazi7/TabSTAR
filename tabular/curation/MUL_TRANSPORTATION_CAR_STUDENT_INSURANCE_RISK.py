from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: insurance_dataset
====
URL: https://www.openml.org/search?type=data&id=43157
====
Description: **Dataset description**

Insurance is a network for evaluating car insurance risks.


**Format of the dataset**


The insurance data set contains the following 27 variables:

GoodStudent (good student): a two-level factor with levels False and True.

Age (age): a three-level factor with levels Adolescent, Adult and Senior.

SocioEcon (socio-economic status): a four-level factor with levels Prole, Middle, UpperMiddle and Wealthy.

RiskAversion (risk aversion): a four-level factor with levels Psychopath, Adventurous, Normal and Cautious.

VehicleYear (vehicle age): a two-level factor with levels Current and older.

ThisCarDam (damage to this car): a four-level factor with levels None, Mild, Moderate and Severe.

RuggedAuto (ruggedness of the car): a three-level factor with levels EggShell, Football and Tank.

Accident (severity of the accident): a four-level factor with levels None, Mild, Moderate and Severe.

MakeModel (car's model): a five-level factor with levels SportsCar, Economy, FamilySedan, Luxury and SuperLuxury.

DrivQuality (driving quality): a three-level factor with levels Poor, Normal and Excellent.

Mileage (mileage): a four-level factor with levels FiveThou, TwentyThou, FiftyThou and Domino.

Antilock (ABS): a two-level factor with levels False and True.

DrivingSkill (driving skill): a three-level factor with levels SubStandard, Normal and Expert.

SeniorTrain (senior training): a two-level factor with levels False and True.

ThisCarCost (costs for the insured car): a four-level factor with levels Thousand, TenThou, HundredThou and Million.

Theft (theft): a two-level factor with levels False and True.

CarValue (value of the car): a five-level factor with levels FiveThou, TenThou, TwentyThou, FiftyThou and Million.

HomeBase (neighbourhood type): a four-level factor with levels Secure, City, Suburb and Rural.

AntiTheft (anti-theft system): a two-level factor with levels False and True.

PropCost (ratio of the cost for the two cars): a four-level factor with levels Thousand, TenThou, HundredThou and Million.

OtherCarCost (costs for the other car): a four-level factor with levels Thousand, TenThou, HundredThou and Million.

OtherCar (other cars involved in the accident): a two-level factor with levels False and True.

MedCost (cost of the medical treatment): a four-level factor with levels Thousand, TenThou, HundredThou and Million.

Cushioning (cushioning): a four-level factor with levels Poor, Fair, Good and Excellent.

Airbag (airbag): a two-level factor with levels False and True.

ILiCost (inspection cost): a four-level factor with levels Thousand, TenThou, HundredThou and Million.

DrivHist (driving history): a three-level factor with levels Zero, One and Many.

**Source **

Binder J, Koller D, Russell S, Kanazawa K (1997). "Adaptive Probabilistic Networks with Hidden Variables". Machine Learning, 29(2-3):213-244.
====
Features:

GoodStudent (nominal, 2 distinct): ['0', '1']
Age (nominal, 3 distinct): ['Adult', 'Adolescent', 'Senior']
SocioEcon (nominal, 4 distinct): ['Prole', 'Middle', 'UpperMiddle', 'Wealthy']
RiskAversion (nominal, 4 distinct): ['Normal', 'Adventurous', 'Cautious', 'Psychopath']
VehicleYear (nominal, 2 distinct): ['Older', 'Current']
ThisCarDam (nominal, 4 distinct): ['None', 'Severe', 'Mild', 'Moderate']
RuggedAuto (nominal, 3 distinct): ['EggShell', 'Football', 'Tank']
Accident (nominal, 4 distinct): ['None', 'Severe', 'Mild', 'Moderate']
MakeModel (nominal, 5 distinct): ['Economy', 'FamilySedan', 'SportsCar', 'Luxury', 'SuperLuxury']
DrivQuality (nominal, 3 distinct): ['Normal', 'Poor', 'Excellent']
Mileage (nominal, 4 distinct): ['FiftyThou', 'TwentyThou', 'Domino', 'FiveThou']
Antilock (nominal, 2 distinct): ['0', '1']
DrivingSkill (nominal, 3 distinct): ['Normal', 'SubStandard', 'Expert']
SeniorTrain (nominal, 2 distinct): ['0', '1']
ThisCarCost (nominal, 4 distinct): ['Thousand', 'TenThou', 'HundredThou', 'Million']
Theft (nominal, 2 distinct): ['0', '1']
CarValue (nominal, 5 distinct): ['FiveThou', 'TwentyThou', 'TenThou', 'FiftyThou', 'Million']
HomeBase (nominal, 4 distinct): ['City', 'Secure', 'Suburb', 'Rural']
AntiTheft (nominal, 2 distinct): ['0', '1']
PropCost (nominal, 4 distinct): ['Thousand', 'TenThou', 'HundredThou', 'Million']
OtherCarCost (nominal, 3 distinct): ['Thousand', 'TenThou', 'HundredThou']
OtherCar (nominal, 2 distinct): ['1', '0']
MedCost (nominal, 4 distinct): ['Thousand', 'TenThou', 'HundredThou', 'Million']
Cushioning (nominal, 4 distinct): ['Poor', 'Fair', 'Good', 'Excellent']
Airbag (nominal, 2 distinct): ['0', '1']
ILiCost (nominal, 4 distinct): ['Thousand', 'TenThou', 'HundredThou', 'Million']
DrivHist (nominal, 3 distinct): ['Zero', 'Many', 'One']
'''

CONTEXT = "Car Insurance Risk Evaluation"
TARGET = CuratedTarget(raw_name="Accident", new_name="Severity of Car Accident", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []
