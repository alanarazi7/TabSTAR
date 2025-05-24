from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: Pollen-Luxembourg-1992-2018
====
Examples: 7784
====
URL: https://www.openml.org/search?type=data&id=43648
====
Description: Daily pollen concentration in luxembourg.
Daily pollen concentration data for 33 pollen types since Jan 1, 1992 in Luxembourg combined with meteo data.

  This is the concentration by m for each type of pollen (graminea, )
  The dataset has been completed with daily meteo data : temperature minimum and maximum in C , precipitation in mm

Data comes from https://data.public.lu/ and http://www.pollen.lu/
**Examples of Critical Thresholds from www.pollen.lu**
Betula, Alnus, Corylus, Quercus, Fagus

Low : 0-10
Medium : 11-50
High : 50 (100)

Gramineae

Low : 0-5
Medium : 6-30
High : 30 (50)

Plantago, Rumex, Chenopodium

Low : 0-3
Medium : 4-15
High : 15

Artemisia

Low : 0-2
Medium : 3-6
High : 6 (20)
====
Features:

Date (string, 7784 distinct): ['1992-01-01', '2010-08-01', '2010-08-29', '2010-08-28', '2010-08-27', '2010-08-26', '2010-08-25', '2010-08-24', '2010-08-23', '2010-08-22']
Ambrosia (numeric, 13 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '11', '8']
Artemisia (numeric, 33 distinct): ['0', '1', '2', '3', '4', '6', '7', '5', '8', '9']
Asteraceae (numeric, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '9', '13', '10']
Alnus (numeric, 145 distinct): ['0', '1', '2', '3', '4', '5', '7', '6', '8', '9']
Betula (numeric, 317 distinct): ['0', '1', '2', '3', '4', '6', '5', '11', '8', '9']
Ericaceae (numeric, 5 distinct): ['0', '1', '2', '3', '4']
Carpinus (numeric, 120 distinct): ['0', '1', '2', '3', '4', '5', '7', '8', '13', '10']
Castanea (numeric, 64 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '7', '10']
Quercus (numeric, 225 distinct): ['0', '1', '2', '3', '4', '6', '5', '7', '8', '11']
Chenopodium (numeric, 11 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '10', '9']
Cupressaceae (numeric, 153 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Acer (numeric, 15 distinct): ['0', '1', '2', '3', '4', '5', '7', '6', '9', '8']
Fraxinus (numeric, 198 distinct): ['0', '1', '2', '3', '4', '5', '8', '7', '6', '9']
Gramineae (numeric, 227 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '10']
Fagus (numeric, 153 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '17']
Juncaceae (numeric, 6 distinct): ['0', '1', '2', '3', '4', '5']
Aesculus (numeric, 13 distinct): ['0', '1', '2', '3', '4', '7', '5', '6', '8', '12']
Larix (numeric, 10 distinct): ['0', '1', '2', '3', '4', '6', '5', '7', '11', '8']
Corylus (numeric, 122 distinct): ['0', '1', '2', '3', '4', '5', '7', '6', '8', '11']
Juglans (numeric, 18 distinct): ['0', '1', '2', '3', '5', '4', '6', '8', '7', '9']
Umbellifereae (numeric, 12 distinct): ['0', '1', '2', '3', '4', '5', '7', '19', '10', '9']
Ulmus (numeric, 31 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7', '9']
Urtica (numeric, 263 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '10']
Rumex (numeric, 29 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Populus (numeric, 42 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Pinaceae (numeric, 207 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
Plantago (numeric, 11 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Platanus (numeric, 46 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
Salix (numeric, 79 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
Cyperaceae (numeric, 6 distinct): ['0', '1', '2', '3', '4', '6']
Filipendula (numeric, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Sambucus (numeric, 27 distinct): ['0', '1', '2', '3', '4', '5', '7', '6', '8', '10']
Tilia (numeric, 20 distinct): ['0', '1', '2', '3', '4', '5', '7', '6', '8', '10']
MaxAirTempC (numeric, 420 distinct): ['17.6', '14.9', '20.2', '20.1', '15.8', '18.5', '15.6', '18.6', '19.9', '17.7']
MinAirTempC (numeric, 347 distinct): ['0.1', '12.6', '10.2', '11.4', '11.9', '7.0', '8.6', '10.1', '8.8', '9.7']
PrecipitationC (numeric, 283 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '0.7', '1.3', '0.9']
'''

CONTEXT = "Pollens in Luxembourg"
TARGET = CuratedTarget(raw_name="Betula", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Date", feat_type=FeatureType.DATE)]