from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: hepatitis_c_virus_hcv_for_egyptian_patients
====
Examples: 1385
====
URL: https://www.openml.org/search?type=data&id=46607
====
Description: Egyptian patients who underwent treatment dosages for HCV about 18 months. Discretization should be applied based on expert recommendations; there is an attached file shows how.

Age	Age
Gender	Gender
BMI	Body Mass Index
Fever	Fever
Nausea/Vomting	Nausea/Vomting
Headache	Headache
Diarrhea	Diarrhea
Fatigue & generalized bone ache	Fatigue & generalized bone ache
Jaundice	Jaundice
Epigastric pain	Epigastric pain
WBC	White blood cell
RBC	red blood cells
HGB	Hemoglobin
Plat	Platelets
AST 1	aspartate transaminase ratio
ALT 1	alanine transaminase ratio 1 week
ALT 4	alanine transaminase ratio 12 weeks
ALT 12	alanine transaminase ratio 4 weeks
ALT 24	alanine transaminase ratio 24 weeks
ALT 36	alanine transaminase ratio 36 weeks
ALT 48	alanine transaminase ratio 48 weeks
ALT after 24 w	alanine transaminase ratio 24 weeks
RNA Base	RNA Base
RNA 4	RNA 4
RNA 12	RNA 12
RNA EOT	RNA end-of-treatment 
RNA EF	RNA Elongation Factor
Baseline histological Grading	Baseline histological Grading
Baselinehistological staging	Baselinehistological staging

A novel model based on non invasive methods for prediction of liver fibrosis
By Mahmoud Nasr, Khaled El-Bahnasy, M. Hamdy, S. Kamal. 2017

Published in International Computer Engineering Conference
====
Target Variable: Baselinehistological_staging (numeric, 4 distinct): ['4', '3', '1', '2']
====
Features:

Age_ (numeric, 30 distinct): ['56', '33', '39', '36', '47', '43', '32', '53', '59', '34']
Gender (numeric, 2 distinct): ['1', '2']
BMI (numeric, 14 distinct): ['34', '24', '28', '33', '31', '35', '27', '23', '22', '30']
Fever (numeric, 2 distinct): ['2', '1']
Nausea/Vomting (numeric, 2 distinct): ['2', '1']
Headache_ (numeric, 2 distinct): ['1', '2']
Diarrhea_ (numeric, 2 distinct): ['2', '1']
Fatigue_&_generalized_bone_ache_ (numeric, 2 distinct): ['1', '2']
Jaundice_ (numeric, 2 distinct): ['2', '1']
Epigastric_pain_ (numeric, 2 distinct): ['2', '1']
WBC (numeric, 1305 distinct): ['3271', '4082', '6038', '3414', '11530', '6569', '7794', '7666', '5878', '5779']
RBC (numeric, 1384 distinct): ['4153369.0', '4248807.0', '3835942.0', '4810099.0', '4393271.0', '4225413.0', '4560429.0', '4275492.0', '4308505.0', '4554941.0']
HGB (numeric, 6 distinct): ['15', '12', '14', '13', '11', '10']
Plat (numeric, 1375 distinct): ['104272.0', '199307.0', '167031.0', '223137.0', '128569.0', '171378.0', '186850.0', '103368.0', '218554.0', '163145.0']
AST_1 (numeric, 90 distinct): ['86', '124', '121', '66', '110', '93', '67', '55', '62', '99']
ALT_1 (numeric, 90 distinct): ['126', '109', '86', '93', '63', '78', '101', '102', '70', '77']
ALT4 (numeric, 90 distinct): ['120', '64', '71', '43', '118', '117', '81', '114', '67', '53']
ALT_12 (numeric, 90 distinct): ['89', '103', '68', '39', '121', '116', '85', '60', '59', '118']
ALT_24 (numeric, 90 distinct): ['115', '101', '126', '84', '107', '122', '104', '41', '81', '47']
ALT_36 (numeric, 91 distinct): ['64', '114', '39', '95', '51', '87', '46', '65', '57', '53']
ALT_48 (numeric, 91 distinct): ['78', '120', '101', '107', '44', '56', '123', '72', '114', '69']
ALT_after_24_w (numeric, 25 distinct): ['34', '25', '30', '24', '45', '43', '36', '33', '28', '41']
RNA_Base (numeric, 1384 distinct): ['23179', '819035', '742700', '1122292', '952626', '1039993', '1002636', '711909', '1033568', '1042810']
RNA_4 (numeric, 1384 distinct): ['709348', '634536', '888527', '339653', '1140348', '1117745', '154321', '589717', '442277', '764396']
RNA_12 (numeric, 1001 distinct): ['5', '288194', '402723', '205762', '404041', '492947', '193601', '31902', '679065', '600276']
RNA_EOT (numeric, 1002 distinct): ['5', '327351', '799675', '307451', '672470', '352946', '543773', '374338', '110417', '621115']
RNA_EF (numeric, 1004 distinct): ['5', '673281', '71909', '112773', '578238', '311999', '530268', '657407', '346312', '354366']
Baseline_histological_Grading (numeric, 14 distinct): ['15', '11', '14', '9', '6', '12', '8', '13', '4', '5']
'''

CONTEXT = "Hepatitis C Virus Prediction for Egyptian Patients"
TARGET = CuratedTarget(raw_name="Baselinehistological_staging", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []