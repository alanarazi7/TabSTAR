from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: regensburg_pediatric_appendicitis
====
Examples: 782
====
URL: https://www.openml.org/search?type=data&id=46603
====
Description: This dataset was acquired in a retrospective study from a cohort of pediatric patients admitted with abdominal pain to Children's Hospital St. Hedwig in Regensburg, Germany. Multiple abdominal B-mode ultrasound images were acquired for most patients, with the number of views varying from 1 to 15. The images depict various regions of interest, such as the abdomen's right lower quadrant, appendix, intestines, lymph nodes and reproductive organs. Alongside multiple US images for each subject, the dataset includes information encompassing laboratory tests, physical examination results, clinical scores, such as Alvarado and pediatric appendicitis scores, and expert-produced ultrasonographic findings. Lastly, the subjects were labeled w.r.t. three target variables: diagnosis (appendicitis vs. no appendicitis), management (surgical vs. conservative) and severity (complicated vs. uncomplicated or no appendicitis). The study was approved by the Ethics Committee of the University of Regensburg (no. 18-1063-101, 18-1063_1-101 and 18-1063_2-101) and was performed following applicable guidelines and regulations.

Diagnosis: [appendicitis, no appendicitis],
Severity: [complicated, uncomplicated],
Management: [conservative, primary surgical, secondary surgical, simultaneous appendectomy]
====
Target Variable: Management (string, 4 distinct): ['conservative', 'primary surgical', 'secondary surgical', 'simultaneous appendectomy']
====
Features:

Age (numeric, 577 distinct): ['11.05', '13.83', '10.9', '11.27', '14.2', '11.96', '11.38', '12.56', '11.37', '11.81']
BMI (numeric, 510 distinct): ['16.0', '17.6', '19.6', '17.3', '17.7', '16.6', '16.9', '18.6', '16.2', '15.3']
Sex (string, 2 distinct): ['male', 'female']
Height (numeric, 187 distinct): ['140.0', '158.0', '165.0', '160.0', '164.0', '163.0', '143.0', '152.0', '151.0', '146.0']
Weight (numeric, 268 distinct): ['50.0', '33.0', '45.0', '54.0', '53.0', '39.0', '40.0', '52.0', '36.0', '56.0']
Length_of_Stay (numeric, 19 distinct): ['3.0', '4.0', '2.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '1.0']
Alvarado_Score (numeric, 11 distinct): ['5.0', '8.0', '7.0', '6.0', '4.0', '9.0', '3.0', '2.0', '10.0', '0.0']
Paedriatic_Appendicitis_Score (numeric, 11 distinct): ['6.0', '4.0', '5.0', '7.0', '3.0', '8.0', '2.0', '9.0', '10.0', '1.0']
Appendix_on_US (string, 2 distinct): ['yes', 'no']
Appendix_Diameter (numeric, 78 distinct): ['8.0', '9.0', '7.0', '6.0', '10.0', '5.0', '11.0', '12.0', '4.0', '7.5']
Migratory_Pain (string, 2 distinct): ['no', 'yes']
Lower_Right_Abd_Pain (string, 2 distinct): ['yes', 'no']
Contralateral_Rebound_Tenderness (string, 2 distinct): ['no', 'yes']
Coughing_Pain (string, 2 distinct): ['no', 'yes']
Nausea (string, 2 distinct): ['yes', 'no']
Loss_of_Appetite (string, 2 distinct): ['yes', 'no']
Body_Temperature (numeric, 46 distinct): ['37.0', '36.8', '37.2', '37.8', '37.4', '37.5', '38.2', '37.3', '38.0', '36.9']
WBC_Count (numeric, 210 distinct): ['8.1', '8.7', '7.0', '6.9', '8.6', '16.1', '8.4', '13.1', '11.0', '8.2']
Neutrophil_Percentage (numeric, 355 distinct): ['79.0', '84.5', '68.4', '84.0', '80.0', '88.0', '87.1', '85.5', '81.8', '75.5']
Segmented_Neutrophils (numeric, 39 distinct): ['63.0', '54.0', '61.0', '59.0', '82.0', '68.0', '62.0', '73.0', '83.0', '72.0']
Neutrophilia (string, 2 distinct): ['no', 'yes']
RBC_Count (numeric, 171 distinct): ['4.93', '4.74', '4.54', '4.73', '4.61', '4.56', '4.59', '4.9', '4.92', '4.77']
Hemoglobin (numeric, 65 distinct): ['13.5', '12.9', '13.2', '13.1', '13.9', '14.0', '12.8', '13.6', '13.8', '12.6']
RDW (numeric, 53 distinct): ['12.6', '12.7', '12.9', '12.8', '12.5', '12.4', '12.1', '12.2', '12.3', '13.1']
Thrombocyte_Count (numeric, 260 distinct): ['234.0', '221.0', '267.0', '275.0', '277.0', '233.0', '305.0', '223.0', '250.0', '245.0']
Ketones_in_Urine (string, 4 distinct): ['no', '+++', '+', '++']
RBC_in_Urine (string, 4 distinct): ['no', '+', '+++', '++']
WBC_in_Urine (string, 4 distinct): ['no', '+', '++', '+++']
CRP (numeric, 146 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '6.0', '5.0', '7.0', '8.0', '15.0']
Dysuria (string, 2 distinct): ['no', 'yes']
Stool (string, 4 distinct): ['normal', 'diarrhea', 'constipation', 'constipation, diarrhea']
Peritonitis (string, 3 distinct): ['no', 'local', 'generalized']
Psoas_Sign (string, 2 distinct): ['no', 'yes']
Ipsilateral_Rebound_Tenderness (string, 2 distinct): ['no', 'yes']
US_Performed (string, 2 distinct): ['yes', 'no']
Free_Fluids (string, 2 distinct): ['no', 'yes']
Appendix_Wall_Layers (string, 4 distinct): ['intact', 'raised', 'partially raised', 'upset']
Target_Sign (string, 2 distinct): ['yes', 'no']
Appendicolith (string, 3 distinct): ['yes', 'no', 'suspected']
Perfusion (string, 4 distinct): ['hyperperfused', 'hypoperfused', 'no', 'present']
Perforation (string, 4 distinct): ['no', 'yes', 'not excluded', 'suspected']
Surrounding_Tissue_Reaction (string, 2 distinct): ['yes', 'no']
Appendicular_Abscess (string, 3 distinct): ['no', 'yes', 'suspected']
Abscess_Location (string, 7 distinct): ['Douglas', 'rechter Unterbauch', 'retrovesikal', 'perityphlitisch', 'an den M. psoas rechts', 'rechter Mittelbauch', 're Mittelbauch']
Pathological_Lymph_Nodes (string, 2 distinct): ['yes', 'no']
Lymph_Nodes_Location (string, 26 distinct): ['mesenterial', 're UB', 'rechter Unterbauch', 'reUB', 'ileocoecal', 're MB', 'rechter Unter- und Mittelbauch', 're UB ', 'ileocöcal', 'MB']
Bowel_Wall_Thickening (string, 2 distinct): ['yes', 'no']
Conglomerate_of_Bowel_Loops (string, 2 distinct): ['no', 'yes']
Ileus (string, 2 distinct): ['no', 'yes']
Coprostasis (string, 2 distinct): ['yes', 'no']
Meteorism (string, 2 distinct): ['yes', 'no']
Enteritis (string, 2 distinct): ['yes', 'no']
Gynecological_Findings (string, 14 distinct): ['keine', 'Ovarialzyste', 'Ovarialzysten', 'Zyste Uterus', 'Ovarialzyste ', 'In beiden Ovarien Zysten darstellbar, links Ovar mit regelrechter Perfusion, rechts etwas vergrößert, keine eindeutige Perfusion nachweisbar. Retrovesikal freie Flüssigkeit mit Binnenecho', 'Ausschluss pathologischer Ovarialbefund', 'kleine Ovarzyste rechts', 'kein Anhalt für eine gynäkologische Ursache der Beschwerden', 'V. a. Ovarialtorsion']
Severity (string, 2 distinct): ['uncomplicated', 'complicated']
Diagnosis (string, 2 distinct): ['appendicitis', 'no appendicitis']
'''

CONTEXT = "Regensburg Pediatric Appendicitis"
TARGET = CuratedTarget(raw_name="Management", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []