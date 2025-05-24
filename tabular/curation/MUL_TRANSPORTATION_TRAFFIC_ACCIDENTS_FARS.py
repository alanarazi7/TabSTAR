from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: fars
====
Examples: 100968
====
URL: https://www.openml.org/search?type=data&id=40672
====
Description: Re-upload of the dataset as it is present in the Penn ML Benchmark (https://github.com/EpistasisLab/penn-ml-benchmarks/tree/master/datasets/classification/fars).
It's a dataset on traffic accidents, see https://data.world/nhtsa/fars-data.
I am not sure of the specific date or aggregation method as it is just a re-upload.
====
Target Variable: class (nominal, 8 distinct): ['1', '4', '2', '5', '6', '7', '3', '0']
====
Features:

CASE_STATE (numeric, 51 distinct): ['4', '43', '9', '10', '32', '33', '38', '13', '22', '35']
AGE (numeric, 99 distinct): ['18', '19', '20', '21', '17', '22', '99', '16', '23', '24']
SEX (nominal, 3 distinct): ['1', '0', '2']
PERSON_TYPE (nominal, 10 distinct): ['1', '6', '7', '0', '8', '2', '5', '3', '4', '9']
SEATING_POSITION (numeric, 26 distinct): ['3', '6', '8', '16', '13', '14', '25', '9', '4', '11']
RESTRAINT_SYSTEM-USE (numeric, 12 distinct): ['7', '5', '11', '8', '4', '6', '1', '10', '9', '2']
AIR_BAG_AVAILABILITY/DEPLOYMENT (numeric, 13 distinct): ['4', '9', '2', '0', '11', '12', '7', '10', '6', '1']
EJECTION (nominal, 4 distinct): ['0', '2', '1', '3']
EJECTION_PATH (nominal, 10 distinct): ['0', '9', '7', '6', '1', '3', '8', '5', '2', '4']
EXTRICATION (nominal, 3 distinct): ['1', '0', '2']
NON_MOTORIST_LOCATION (numeric, 18 distinct): ['16', '11', '10', '12', '9', '0', '2', '4', '13', '3']
POLICE_REPORTED_ALCOHOL_INVOLVEMENT (nominal, 4 distinct): ['1', '0', '2', '3']
METHOD_ALCOHOL_DETERMINATION (nominal, 7 distinct): ['2', '1', '3', '4', '6', '0', '5']
ALCOHOL_TEST_TYPE (nominal, 10 distinct): ['4', '9', '6', '2', '5', '7', '8', '1', '3', '0']
ALCOHOL_TEST_RESULT (numeric, 69 distinct): ['96', '0', '99', '97', '16', '17', '20', '19', '14', '15']
POLICE-REPORTED_DRUG_INVOLVEMENT (nominal, 4 distinct): ['2', '1', '3', '0']
METHOD_OF_DRUG_DETERMINATION (nominal, 5 distinct): ['3', '4', '2', '0', '1']
DRUG_TEST_TYPE (nominal, 7 distinct): ['2', '0', '5', '6', '1', '4', '3']
DRUG_TEST_RESULTS_(1_of_3) (numeric, 95 distinct): ['0.0', '999.0', '1.0', '997.0', '996.0', '407.0', '695.0', '401.0', '402.0', '603.0']
DRUG_TEST_TYPE_(2_of_3) (nominal, 7 distinct): ['2', '5', '0', '6', '1', '3', '4']
DRUG_TEST_RESULTS_(2_of_3) (numeric, 73 distinct): ['0.0', '999.0', '1.0', '996.0', '417.0', '407.0', '606.0', '402.0', '351.0', '695.0']
DRUG_TEST_TYPE_(3_of_3) (nominal, 7 distinct): ['2', '5', '0', '6', '1', '3', '4']
DRUG_TEST_RESULTS_(3_of_3) (numeric, 59 distinct): ['0.0', '999.0', '1.0', '996.0', '695.0', '606.0', '417.0', '351.0', '407.0', '410.0']
HISPANIC_ORIGIN (nominal, 9 distinct): ['6', '5', '8', '4', '3', '0', '7', '1', '2']
TAKEN_TO_HOSPITAL (nominal, 3 distinct): ['2', '0', '1']
RELATED_FACTOR_(1)-PERSON_LEVEL (numeric, 45 distinct): ['27', '15', '43', '2', '28', '8', '41', '17', '7', '18']
RELATED_FACTOR_(2)-PERSON_LEVEL (numeric, 48 distinct): ['29', '12', '18', '44', '46', '4', '20', '10', '30', '33']
RELATED_FACTOR_(3)-PERSON_LEVEL (numeric, 33 distinct): ['19', '30', '8', '7', '20', '12', '14', '31', '11', '29']
RACE (numeric, 18 distinct): ['11', '17', '15', '4', '1', '6', '0', '12', '5', '16']
'''

CONTEXT = "Traffic Accidents"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []