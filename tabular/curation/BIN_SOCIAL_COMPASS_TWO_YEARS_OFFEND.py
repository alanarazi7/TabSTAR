from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: compas-two-years
====
Examples: 4966
====
URL: https://www.openml.org/search?type=data&id=45039
====
Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark, transformed in the same way. This dataset belongs to the "classification on both numerical and categorical features" benchmark. 
 
  Original link: https://openml.org/d/42192 
 
 Original description: 
 
nominal features and target for COMPAS
====
Target Variable: twoyearrecid (nominal, 2 distinct): ['0', '1']
====
Features:

sex (nominal, 2 distinct): ['1', '0']
age (numeric, 61 distinct): ['24', '26', '27', '22', '25', '21', '23', '30', '28', '29']
juv_misd_count (numeric, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '12', '13']
priors_count (numeric, 36 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
age_cat_25-45 (nominal, 2 distinct): ['1', '0']
age_cat_Greaterthan45 (nominal, 2 distinct): ['0', '1']
age_cat_Lessthan25 (nominal, 2 distinct): ['0', '1']
race_African-American (nominal, 2 distinct): ['1', '0']
race_Caucasian (nominal, 2 distinct): ['0', '1']
c_charge_degree_F (nominal, 2 distinct): ['1', '0']
c_charge_degree_M (nominal, 2 distinct): ['0', '1']
'''

CONTEXT = "Correctional Offender Management Profiling for Alternative Sanctions (COMPASS): Two Years Recidivism"
TARGET = CuratedTarget(raw_name="twoyearrecid", new_name="Two Year Recidivism", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'No', '1': 'Yes'})
COLS_TO_DROP = []
FEATURES = []