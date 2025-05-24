from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cdc_diabetes
====
Examples: 253680
====
URL: https://www.openml.org/search?type=data&id=46598
====
Description: For what purpose was the dataset created?

To better understand the relationship between  lifestyle and diabetes in the US

Who funded the creation of the dataset?

The CDC

What do the instances in this dataset represent?

Each row represents a person participating in this study.

Are there recommended data splits?

Cross validation or a fixed train-test split could be used.

Does the dataset contain data that might be considered sensitive in any way?

- Gender
- Income
- Education level

Was there any data preprocessing performed?

Bucketing of age

Additional Information

Dataset link: https://www.cdc.gov/brfss/annual_data/annual_2014.html

Has Missing Values?

No

- Diabetes diagnosis
- Demographics (race, sex)
- Personal information (income, educations)
- Health history (drinking, smoking, mental health, physical health)

Class Labels

- Diabetes
- Pre-diabetes
- Healthy
====
Target Variable: Diabetes_binary (numeric, 2 distinct): ['0', '1']
====
Features:

HighBP (numeric, 2 distinct): ['0', '1']
HighChol (numeric, 2 distinct): ['0', '1']
CholCheck (numeric, 2 distinct): ['1', '0']
BMI (numeric, 84 distinct): ['27', '26', '24', '25', '28', '23', '29', '30', '22', '31']
Smoker (numeric, 2 distinct): ['0', '1']
Stroke (numeric, 2 distinct): ['0', '1']
HeartDiseaseorAttack (numeric, 2 distinct): ['0', '1']
PhysActivity (numeric, 2 distinct): ['1', '0']
Fruits (numeric, 2 distinct): ['1', '0']
Veggies (numeric, 2 distinct): ['1', '0']
HvyAlcoholConsump (numeric, 2 distinct): ['0', '1']
AnyHealthcare (numeric, 2 distinct): ['1', '0']
NoDocbcCost (numeric, 2 distinct): ['0', '1']
GenHlth (numeric, 5 distinct): ['2', '3', '1', '4', '5']
MentHlth (numeric, 31 distinct): ['0', '2', '30', '5', '1', '3', '10', '15', '4', '20']
PhysHlth (numeric, 31 distinct): ['0', '30', '2', '1', '3', '5', '10', '15', '4', '7']
DiffWalk (numeric, 2 distinct): ['0', '1']
Sex (numeric, 2 distinct): ['0', '1']
Age (numeric, 13 distinct): ['9', '10', '8', '7', '11', '6', '13', '5', '12', '4']
Education (numeric, 6 distinct): ['6', '5', '4', '3', '2', '1']
Income (numeric, 8 distinct): ['8', '7', '6', '5', '4', '3', '2', '1']
'''

CONTEXT = "CDC Diabetes Lifestyle Study"
TARGET = CuratedTarget(raw_name='Diabetes_binary', new_name='Diabetes Outcome', task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []