from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Diabetes_Dataset
====
Examples: 768
====
URL: https://www.openml.org/search?type=data&id=46254
====
Description: Description:
The dataset, named 'diabetes.csv', serves as a comprehensive resource for understanding various factors that may influence the occurrence of diabetes in individuals. Consisting of several medically relevant parameters, the dataset captures key details across 9 columns, namely Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI (Body Mass Index), DiabetesPedigreeFunction, Age, and Outcome. Each column reflects a distinct attribute significant to diabetes research and potential predictive modeling.

Attribute Description:
1. Pregnancies: Number of times pregnant (Example values: 2, 1)
2. Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test (Example values: 82, 142)
3. BloodPressure: Diastolic blood pressure (mm Hg) (Example values: 70, 64)
4. SkinThickness: Triceps skin fold thickness (mm) (Example values: 27, 0)
5. Insulin: 2-Hour serum insulin (mu U/ml) (Example values: 168, 0)
6. BMI: Body mass index (weight in kg/(height in m)^2) (Example values: 36.8, 30.1)
7. DiabetesPedigreeFunction: Diabetes pedigree function (Example values: 0.34, 0.396)
8. Age: Age in years (Example values: 54, 24)
9. Outcome: Class variable (0 or 1) where 1 denotes the presence of diabetes and 0 denotes absence (Example values: 1, 0)

Use Case:
This dataset is particularly useful for medical researchers, data scientists, and healthcare providers seeking to identify patterns or factors that significantly contribute to diabetes. By employing statistical analysis or machine learning models, one can predict the likelihood of diabetes occurrence based on the dataset's parameters. Furthermore, this dataset can facilitate a better understanding of how various factors, such as pregnancy, BMI, and age, interact with each other in the context of diabetes, thereby aiding in preventative healthcare planning and patient education.
====
Features:

Pregnancies (numeric, 17 distinct): ['1', '0', '2', '3', '4', '5', '6', '7', '8', '9']
Glucose (numeric, 136 distinct): ['99', '100', '111', '129', '125', '106', '112', '108', '95', '105']
BloodPressure (numeric, 47 distinct): ['70', '74', '78', '68', '72', '64', '80', '76', '60', '0']
SkinThickness (numeric, 51 distinct): ['0', '32', '30', '27', '23', '33', '28', '18', '31', '19']
Insulin (numeric, 186 distinct): ['0', '105', '130', '140', '120', '94', '180', '100', '135', '115']
BMI (numeric, 248 distinct): ['32.0', '31.6', '31.2', '0.0', '32.4', '33.3', '30.1', '32.8', '32.9', '30.8']
DiabetesPedigreeFunction (numeric, 517 distinct): ['0.258', '0.254', '0.268', '0.207', '0.261', '0.259', '0.238', '0.19', '0.263', '0.299']
Age (numeric, 52 distinct): ['22', '21', '25', '24', '23', '28', '26', '27', '29', '31']
Outcome (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Diabetes Risk Factors"
TARGET = CuratedTarget(raw_name="Outcome", new_name="Diabetes Outcome", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []