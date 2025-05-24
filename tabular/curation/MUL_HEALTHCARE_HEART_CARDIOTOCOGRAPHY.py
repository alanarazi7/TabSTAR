from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cardiotocography
====
Examples: 2126
====
URL: https://www.openml.org/search?type=data&id=1466
====
Description: **Author**: J. P. Marques de Sá, J. Bernardes, D. Ayers de Campos.  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Cardiotocography)  
**Please cite**: Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318, [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.

### Attribute Information:  
LB - FHR baseline (beats per minute)  
AC - # of accelerations per second  
FM - # of fetal movements per second  
UC - # of uterine contractions per second  
DL - # of light decelerations per second  
DS - # of severe decelerations per second  
DP - # of prolongued decelerations per second  
ASTV - percentage of time with abnormal short term variability  
MSTV - mean value of short term variability  
ALTV - percentage of time with abnormal long term variability  
MLTV - mean value of long term variability  
Width - width of FHR histogram  
Min - minimum of FHR histogram  
Max - Maximum of FHR histogram  
Nmax - # of histogram peaks  
Nzeros - # of histogram zeros  
Mode - histogram mode  
Mean - histogram mean  
Median - histogram median  
Variance - histogram variance  
Tendency - histogram tendency  
CLASS - FHR pattern class code (1 to 10)  
NSP - fetal state class code (N=normal; S=suspect; P=pathologic)  

### Relevant Papers:
Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318
====
Target Variable: Class (nominal, 10 distinct): ['2', '1', '6', '7', '10', '8', '4', '5', '9', '3']
====
Features:

V1 (numeric, 48 distinct): ['30', '44', '19', '46', '45', '10', '1', '4', '3', '36']
V2 (numeric, 979 distinct): ['0.0', '8.0', '12.0', '10.0', '30.0', '16.0', '17.0', '21.0', '25.0', '14.0']
V3 (numeric, 1064 distinct): ['1199.0', '3599.0', '3540.0', '1192.0', '1185.0', '1182.0', '1194.0', '1189.0', '1174.0', '1014.0']
V4 (numeric, 48 distinct): ['133', '130', '122', '138', '125', '128', '120', '142', '144', '132']
V5 (numeric, 48 distinct): ['133', '130', '122', '138', '125', '128', '120', '142', '144', '132']
V6 (numeric, 22 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
V7 (numeric, 96 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '6.0', '7.0', '5.0', '8.0', '10.0']
V8 (numeric, 19 distinct): ['0', '3', '4', '1', '2', '5', '6', '7', '8', '9']
V9 (numeric, 75 distinct): ['60', '58', '65', '63', '64', '61', '51', '62', '22', '25']
V10 (numeric, 57 distinct): ['0.8', '1.3', '0.5', '0.4', '0.7', '0.9', '0.6', '1.2', '1.5', '1.0']
V11 (numeric, 87 distinct): ['0', '1', '2', '5', '4', '3', '8', '6', '12', '10']
V12 (numeric, 249 distinct): ['0.0', '7.1', '6.7', '6.5', '5.2', '9.5', '6.8', '5.6', '7.2', '8.5']
V13 (numeric, 15 distinct): ['0', '1', '2', '4', '3', '5', '6', '7', '8', '9']
V14 (numeric, 2 distinct): ['0', '1']
V15 (numeric, 5 distinct): ['0', '1', '2', '3', '4']
V16 (numeric, 154 distinct): ['39', '102', '27', '31', '90', '98', '96', '83', '22', '42']
V17 (numeric, 109 distinct): ['50', '52', '71', '120', '60', '68', '67', '103', '51', '62']
V18 (numeric, 86 distinct): ['157', '171', '158', '156', '159', '152', '154', '178', '172', '153']
V19 (numeric, 18 distinct): ['1', '2', '3', '4', '5', '6', '7', '0', '8', '9']
V20 (numeric, 9 distinct): ['0', '1', '2', '3', '4', '5', '10', '8', '7']
V21 (numeric, 88 distinct): ['133', '136', '150', '142', '148', '144', '129', '143', '125', '126']
V22 (numeric, 103 distinct): ['143', '144', '135', '141', '140', '132', '133', '145', '136', '147']
V23 (numeric, 95 distinct): ['146', '137', '142', '145', '147', '151', '141', '134', '149', '143']
V24 (numeric, 133 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '8.0', '6.0', '7.0', '9.0']
V25 (numeric, 3 distinct): ['0.0', '1.0', '-1.0']
V26 (numeric, 2 distinct): ['0', '1']
V27 (numeric, 2 distinct): ['0', '1']
V28 (numeric, 2 distinct): ['0', '1']
V29 (numeric, 2 distinct): ['0', '1']
V30 (numeric, 2 distinct): ['0', '1']
V31 (numeric, 2 distinct): ['0', '1']
V32 (numeric, 2 distinct): ['0', '1']
V33 (numeric, 2 distinct): ['0', '1']
V34 (numeric, 2 distinct): ['0', '1']
V35 (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Cardiotocography - Fetal Cardiotocograms"
TARGET = CuratedTarget(raw_name="CLASS", new_name="FHR Pattern Class Code", task_type=SupervisedTask.MULTICLASS)
# These all seem like labels
COLS_TO_DROP = ["V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35"]
FEATURES = []