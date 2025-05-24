from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Heart_Failure_Prediction
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=45950
====
Description: Description:
This dataset, named "heart_failure_clinical_records.csv", consists of clinical records of patients with heart failure. It includes various attributes such as age, anaemia, creatinine phosphokinase levels, diabetes status, ejection fraction, high blood pressure presence, platelet count, serum creatinine levels, serum sodium levels, gender, smoking habits, follow-up time, and a binary indicator for death event.

Attribute Description:
- age: age of the patient
- anaemia: presence of anaemia (0: no, 1: yes)
- creatinine_phosphokinase: level of creatinine phosphokinase in the blood
- diabetes: presence of diabetes (0: no, 1: yes)
- ejection_fraction: percentage of blood leaving the heart at each contraction
- high_blood_pressure: presence of high blood pressure (0: no, 1: yes)
- platelets: platelet count in the blood
- serum_creatinine: level of serum creatinine in the blood
- serum_sodium: level of serum sodium in the blood
- sex: gender of the patient (0: female, 1: male)
- smoking: smoking status of the patient (0: no, 1: yes)
- time: follow-up time
- DEATH_EVENT: indicator of death occurence during the follow-up period (0: no, 1: yes)

Use Case:
This dataset can be used for analyzing the relationship between various clinical attributes and the occurrence of death events in patients with heart failure. It can help in predicting the risk factors associated with heart failure mortality.
====
Target Variable: DEATH_EVENT (numeric, 2 distinct): ['0', '1']
====
Features:

age (numeric, 48 distinct): ['60.0', '50.0', '65.0', '70.0', '45.0', '55.0', '53.0', '58.0', '75.0', '42.0']
anaemia (numeric, 2 distinct): ['0', '1']
creatinine_phosphokinase (numeric, 290 distinct): ['582', '66', '129', '102', '68', '64', '59', '47', '135', '84']
diabetes (numeric, 2 distinct): ['0', '1']
ejection_fraction (numeric, 17 distinct): ['35', '40', '38', '30', '25', '60', '45', '50', '20', '55']
high_blood_pressure (numeric, 2 distinct): ['0', '1']
platelets (numeric, 203 distinct): ['263358.03', '226000.0', '255000.0', '305000.0', '237000.0', '302000.0', '127000.0', '362000.0', '271000.0', '283000.0']
serum_creatinine (numeric, 43 distinct): ['1.0', '1.1', '0.9', '1.2', '0.8', '0.7', '1.3', '1.18', '1.7', '1.6']
serum_sodium (numeric, 27 distinct): ['137', '140', '136', '134', '138', '139', '141', '135', '145', '132']
sex (numeric, 2 distinct): ['1', '0']
smoking (numeric, 2 distinct): ['0', '1']
time (numeric, 155 distinct): ['74', '30', '187', '10', '244', '95', '186', '88', '214', '245']
'''

CONTEXT = "Clinical Records of Patients with Heart Failure"
TARGET = CuratedTarget(raw_name="DEATH_EVENT", new_name="Death Event due to Heart Failure",
                       task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="sex", value_mapping={"0": "Female", "1": "Male"}),
            CuratedFeature(raw_name="time", new_name="Follow-up Time")]
