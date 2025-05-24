from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Estimation_of_Obesity_Levels
====
Examples: 2111
====
URL: https://www.openml.org/search?type=data&id=46597
====
Description: This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77 percentage of the data was generated synthetically using the Weka tool and the SMOTE filter, 23 percentage of the data was collected directly from users through a web platform.

Read the article (https://doi.org/10.1016/j.dib.2019.104344) to see the description of the attributes.

Class Labels

Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III
====
Target Variable: NObeyesdad (string, 7 distinct): ['Obesity_Type_I', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_I', 'Overweight_Level_II', 'Normal_Weight', 'Insufficient_Weight']
====
Features:

Gender (string, 2 distinct): ['Male', 'Female']
Age (numeric, 1402 distinct): ['18.0', '26.0', '21.0', '23.0', '19.0', '20.0', '22.0', '17.0', '24.0', '25.0']
Height (numeric, 1574 distinct): ['1.7', '1.65', '1.6', '1.75', '1.62', '1.8', '1.72', '1.63', '1.67', '1.78']
Weight (numeric, 1525 distinct): ['80.0', '70.0', '50.0', '75.0', '60.0', '65.0', '42.0', '90.0', '78.0', '45.0']
family_history_with_overweight (string, 2 distinct): ['yes', 'no']
FAVC (string, 2 distinct): ['yes', 'no']
FCVC (numeric, 810 distinct): ['3.0', '2.0', '1.0', '2.8232', '2.215', '2.7951', '2.4425', '2.8165', '2.938', '2.955']
NCP (numeric, 635 distinct): ['3.0', '1.0', '4.0', '2.7768', '3.9854', '1.7376', '1.8944', '1.1046', '2.6447', '3.5598']
CAEC (string, 4 distinct): ['Sometimes', 'Frequently', 'Always', 'no']
SMOKE (string, 2 distinct): ['no', 'yes']
CH2O (numeric, 1268 distinct): ['2.0', '1.0', '3.0', '2.8256', '1.6363', '2.116', '2.1742', '2.53', '2.4501', '1.44']
SCC (string, 2 distinct): ['no', 'yes']
FAF (numeric, 1190 distinct): ['0.0', '1.0', '2.0', '3.0', '0.1102', '1.6616', '0.2454', '1.0678', '0.288', '1.2525']
TUE (numeric, 1129 distinct): ['0.0', '1.0', '2.0', '0.6309', '1.1199', '0.0026', '0.0093', '0.8324', '1.3659', '0.8285']
CALC (string, 4 distinct): ['Sometimes', 'no', 'Frequently', 'Always']
MTRANS (string, 5 distinct): ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike']
'''

CONTEXT = "Obesity Levels Estimation"
TARGET = CuratedTarget(raw_name="NObeyesdad", new_name="Obesity Level", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []