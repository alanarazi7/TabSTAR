from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Cardiovascular-Disease-dataset
====
Examples: 70000
====
URL: https://www.openml.org/search?type=data&id=45547
====
Description: ## Data description

There are 3 types of input features:

* Objective: factual information;
* Examination: results of medical examination;
* Subjective: information given by the patient.

Features:

1. Age | Objective Feature | age | int (days)
2. Height | Objective Feature | height | int (cm) |
3. Weight | Objective Feature | weight | float (kg) |
4. Gender | Objective Feature | gender | categorical code |
5. Systolic blood pressure | Examination Feature | ap_hi | int |
6. Diastolic blood pressure | Examination Feature | ap_lo | int |
7. Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
8. Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
9. Smoking | Subjective Feature | smoke | binary |
10. Alcohol intake | Subjective Feature | alco | binary |
11. Physical activity | Subjective Feature | active | binary |
12. Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

All of the dataset values were collected at the moment of medical examination.

**Notes by Uploader to OpenML**

* Gender: 1 - women, 2 - men
* There is no information available on Kaggle where this data was collected.
====
Target Variable: cardio (nominal, 2 distinct): ['0', '1']
====
Features:

age (numeric, 8076 distinct): ['19741', '18236', '20376', '18253', '20442', '20464', '18184', '20457', '21159', '21892']
gender (nominal, 2 distinct): ['1', '2']
height (numeric, 109 distinct): ['165', '160', '170', '168', '164', '158', '162', '169', '156', '167']
weight (numeric, 287 distinct): ['65.0', '70.0', '68.0', '75.0', '60.0', '80.0', '72.0', '69.0', '78.0', '74.0']
ap_hi (numeric, 153 distinct): ['120.0', '140.0', '130.0', '110.0', '150.0', '160.0', '100.0', '90.0', '170.0', '180.0']
ap_lo (numeric, 157 distinct): ['80.0', '90.0', '70.0', '100.0', '60.0', '1000.0', '110.0', '79.0', '85.0', '75.0']
cholesterol (nominal, 3 distinct): ['1', '2', '3']
gluc (nominal, 3 distinct): ['1', '3', '2']
smoke (nominal, 2 distinct): ['0', '1']
alco (nominal, 2 distinct): ['0', '1']
active (nominal, 2 distinct): ['1', '0']
'''

CONTEXT = "Patient Medical Examination Records for Cardiovascular Disease"
TARGET = CuratedTarget(raw_name="cardio", new_name="Cardiovascular Disease", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
