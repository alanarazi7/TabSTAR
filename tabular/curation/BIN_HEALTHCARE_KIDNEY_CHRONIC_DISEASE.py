from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: chronic-kidney-disease
====
Examples: 400
====
URL: https://www.openml.org/search?type=data&id=42972
====
Description: **Author**: L.Jerlin Rubini
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) - 2015
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  

**Chronic_Kidney_Disease Data Set**

This dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period.

### Attribute information

We use 24 + class = 25 ( 11 numeric ,14 nominal) 
1. Age(numerical) 
age in years 
2. Blood Pressure(numerical) 
bp in mm/Hg 
3. Specific Gravity(nominal) 
4. Albumin(nominal) 
5. Sugar(nominal) 
6. Red Blood Cells(nominal) 
7. Pus Cell (nominal) 
8. Pus Cell clumps(nominal) 
9. Bacteria(nominal) 
10. Blood Glucose Random(numerical) 
11.Blood Urea(numerical) 
12. Serum Creatinine(numerical) 
13. Sodium(numerical) 
14. Potassium(numerical) 
15. Hemoglobin(numerical) 
16.Packed Cell Volume(numerical) 
17. White Blood Cell Count(numerical) 
18. Red Blood Cell Count(numerical) 
19. Hypertension(nominal) 
20. Diabetes Mellitus(nominal) 
21. Coronary Artery Disease(nominal) 
22. Appetite(nominal) 
23. Pedal Edema(nominal) 
24. Anemia(nominal) 
25. Class (nominal)
====
Features:

id (numeric, 400 distinct): ['0', '263', '273', '272', '271', '270', '269', '268', '267', '266']
age (numeric, 76 distinct): ['60.0', '65.0', '48.0', '55.0', '50.0', '47.0', '62.0', '56.0', '59.0', '54.0']
bp (numeric, 10 distinct): ['80.0', '70.0', '60.0', '90.0', '100.0', '50.0', '110.0', '140.0', '180.0', '120.0']
sg (numeric, 5 distinct): ['1.02', '1.01', '1.025', '1.015', '1.005']
al (numeric, 6 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0']
su (numeric, 6 distinct): ['0.0', '2.0', '3.0', '4.0', '1.0', '5.0']
rbc (string, 2 distinct): ['normal', 'abnormal']
pc (string, 2 distinct): ['normal', 'abnormal']
pcc (string, 2 distinct): ['notpresent', 'present']
ba (string, 2 distinct): ['notpresent', 'present']
bgr (numeric, 146 distinct): ['99.0', '93.0', '100.0', '107.0', '131.0', '140.0', '130.0', '92.0', '117.0', '109.0']
bu (numeric, 118 distinct): ['46.0', '25.0', '19.0', '40.0', '50.0', '15.0', '48.0', '18.0', '26.0', '49.0']
sc (numeric, 84 distinct): ['1.2', '1.1', '0.5', '1.0', '0.9', '0.7', '0.6', '0.8', '2.2', '1.5']
sod (numeric, 34 distinct): ['135.0', '140.0', '141.0', '139.0', '138.0', '142.0', '137.0', '150.0', '136.0', '147.0']
pot (numeric, 40 distinct): ['3.5', '5.0', '4.9', '4.7', '4.8', '3.9', '3.8', '4.1', '4.2', '4.0']
hemo (numeric, 115 distinct): ['15.0', '10.9', '13.6', '13.0', '9.8', '11.1', '10.3', '11.3', '13.9', '12.0']
pcv (string, 44 distinct): ['41', '52', '44', '48', '40', '43', '42', '45', '32', '36']
wc (string, 92 distinct): ['9800', '6700', '9200', '9600', '7200', '5800', '6900', '11000', '7800', '9400']
rc (string, 49 distinct): ['5.2', '4.5', '4.9', '4.7', '4.8', '3.9', '4.6', '3.4', '5.9', '5.5']
htn (string, 2 distinct): ['no', 'yes']
dm (string, 5 distinct): ['no', 'yes', '\tno', '\tyes', ' yes']
cad (string, 3 distinct): ['no', 'yes', '\tno']
appet (string, 2 distinct): ['good', 'poor']
pe (string, 2 distinct): ['no', 'yes']
ane (string, 2 distinct): ['no', 'yes']
classification (string, 3 distinct): ['ckd', 'notckd', 'ckd\t']
'''


CONTEXT = "Kidney Chronic Disease"
TARGET = CuratedTarget(raw_name="classification", new_name="Kidney Classification", task_type=SupervisedTask.BINARY,
                       label_mapping={"notckd": "No Kidney Disease",
                                      "ckd": "Kidney Disease",
                                      "ckd\t": "Kidney Disease"})
COLS_TO_DROP = []
FEATURES = []