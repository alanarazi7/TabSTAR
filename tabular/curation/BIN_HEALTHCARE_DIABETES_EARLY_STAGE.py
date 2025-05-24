from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Early-Stage-Diabetes-Risk-Prediction-Dataset
====
Examples: 520
====
URL: https://www.openml.org/search?type=data&id=43643
====
Description: Data Set Information:
This has been collected using direct questionnaires from the patients of Sylhet Diabetes
Hospital in Sylhet, Bangladesh and approved by a doctor.
Data Set Information:
This has been col-
lected using direct questionnaires from the patients of Sylhet Diabetes
Hospital in Sylhet, Bangladesh and approved by a doctor.
Attribute Information:
Age 1.20-65
Sex 1. Male, 2.Female
Polyuria 1.Yes, 2.No.
Polydipsia 1.Yes, 2.No.
sudden weight loss 1.Yes, 2.No.
weakness 1.Yes, 2.No.
Polyphagia 1.Yes, 2.No.
Genital thrush 1.Yes, 2.No.
visual blurring 1.Yes, 2.No.
Itching 1.Yes, 2.No.
Irritability 1.Yes, 2.No.
delayed healing 1.Yes, 2.No.
partial paresis 1.Yes, 2.No.
muscle stiness 1.Yes, 2.No.
Alopecia 1.Yes, 2.No.
Obesity 1.Yes, 2.No.
Class 1.Positive, 2.Negative.
Relevant Papers:
Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques
[Web Link]
Authors and affiliations
M. M. Faniqul IslamEmail
Rahatara Ferdousi
Sadikur Rahman
Humayra Yasmin Bushra
Citation Request:
Islam, MM Faniqul, et al. 'Likelihood prediction of diabetes at early stage using data mining techniques.' Computer Vision and Machine Intelligence in Medical Image Analysis. Springer, Singapore, 2020. 113-125.
Islam, MM Faniqul, et al. 'Likelihood prediction of diabetes at early stage using data mining techniques.' Computer Vision and Machine Intelligence in Medical Image Analysis. Springer, Singapore, 2020. 113-125.
====
Features:

Age (numeric, 51 distinct): ['35', '48', '30', '43', '40', '55', '47', '38', '53', '45']
Gender (string, 2 distinct): ['Male', 'Female']
Polyuria (string, 2 distinct): ['No', 'Yes']
Polydipsia (string, 2 distinct): ['No', 'Yes']
sudden_weight_loss (string, 2 distinct): ['No', 'Yes']
weakness (string, 2 distinct): ['Yes', 'No']
Polyphagia (string, 2 distinct): ['No', 'Yes']
Genital_thrush (string, 2 distinct): ['No', 'Yes']
visual_blurring (string, 2 distinct): ['No', 'Yes']
Itching (string, 2 distinct): ['No', 'Yes']
Irritability (string, 2 distinct): ['No', 'Yes']
delayed_healing (string, 2 distinct): ['No', 'Yes']
partial_paresis (string, 2 distinct): ['No', 'Yes']
muscle_stiffness (string, 2 distinct): ['No', 'Yes']
Alopecia (string, 2 distinct): ['No', 'Yes']
Obesity (string, 2 distinct): ['No', 'Yes']
class (string, 2 distinct): ['Positive', 'Negative']
'''

CONTEXT = "Diabetes Early Stage Risk Prediction in Bangladesh"
TARGET = CuratedTarget(raw_name="class", new_name="Diabetes Risk", task_type=SupervisedTask.BINARY,
                       label_mapping={"Positive": "Positive Risk", "Negative": "Negative Risk"})
COLS_TO_DROP = []
FEATURES = []