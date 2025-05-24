from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ibm-employee-attrition
====
Examples: 1470
====
URL: https://www.openml.org/search?type=data&id=43893
====
Description: IBM Employee Attrition Data 
 The dataset used in the code pattern is supplied by Kaggle and contains HR analytics data of employees
that stay and leave. The types of data include metrics such as education level, job satisfactions, and commmute distance. 
 The dataset was obtained
from https://github.com/IBM/employee-attrition-aif360. 

The dataset is available under the Open Dataset License and the Database Content License.
====
Target Variable: Attrition (string, 2 distinct): ['No', 'Yes']
====
Features:

Age (numeric, 43 distinct): ['35', '34', '36', '31', '29', '32', '30', '33', '38', '40']
BusinessTravel (string, 3 distinct): ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
DailyRate (numeric, 886 distinct): ['691.0', '408.0', '530.0', '1329.0', '1082.0', '329.0', '829.0', '1469.0', '267.0', '217.0']
Department (string, 3 distinct): ['Research & Development', 'Sales', 'Human Resources']
DistanceFromHome (numeric, 29 distinct): ['2', '1', '10', '9', '3', '7', '8', '5', '4', '6']
Education (numeric, 5 distinct): ['3', '4', '2', '1', '5']
EducationField (string, 6 distinct): ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']
EmployeeCount (numeric, 1 distinct): ['1']
EmployeeNumber (numeric, 1470 distinct): ['1.0', '1391.0', '1389.0', '1387.0', '1383.0', '1382.0', '1380.0', '1379.0', '1377.0', '1375.0']
EnvironmentSatisfaction (numeric, 4 distinct): ['3', '4', '2', '1']
Gender (string, 2 distinct): ['Male', 'Female']
HourlyRate (numeric, 71 distinct): ['66', '98', '42', '48', '84', '57', '79', '96', '54', '52']
JobInvolvement (numeric, 4 distinct): ['3', '2', '4', '1']
JobLevel (numeric, 5 distinct): ['1', '2', '3', '4', '5']
JobRole (string, 9 distinct): ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']
JobSatisfaction (numeric, 4 distinct): ['4', '3', '1', '2']
MaritalStatus (string, 3 distinct): ['Married', 'Single', 'Divorced']
MonthlyIncome (numeric, 1349 distinct): ['2342.0', '6142.0', '2741.0', '2559.0', '2610.0', '2451.0', '5562.0', '3452.0', '2380.0', '6347.0']
MonthlyRate (numeric, 1427 distinct): ['4223.0', '9150.0', '9558.0', '12858.0', '22074.0', '25326.0', '9096.0', '13008.0', '12355.0', '7744.0']
NumCompaniesWorked (numeric, 10 distinct): ['1', '0', '3', '2', '4', '7', '6', '5', '9', '8']
Over18 (string, 1 distinct): ['Y']
OverTime (string, 2 distinct): ['No', 'Yes']
PercentSalaryHike (numeric, 15 distinct): ['11', '13', '14', '12', '15', '18', '17', '16', '19', '22']
PerformanceRating (numeric, 2 distinct): ['3', '4']
RelationshipSatisfaction (numeric, 4 distinct): ['3', '4', '2', '1']
StandardHours (numeric, 1 distinct): ['80']
StockOptionLevel (numeric, 4 distinct): ['0', '1', '2', '3']
TotalWorkingYears (numeric, 40 distinct): ['10', '6', '8', '9', '5', '7', '1', '4', '12', '3']
TrainingTimesLastYear (numeric, 7 distinct): ['2', '3', '4', '5', '1', '6', '0']
WorkLifeBalance (numeric, 4 distinct): ['3', '2', '4', '1']
YearsAtCompany (numeric, 37 distinct): ['5', '1', '3', '2', '10', '4', '7', '9', '8', '6']
YearsInCurrentRole (numeric, 19 distinct): ['2', '0', '7', '3', '4', '8', '9', '1', '6', '5']
YearsSinceLastPromotion (numeric, 16 distinct): ['0', '1', '2', '7', '4', '3', '5', '6', '11', '8']
YearsWithCurrManager (numeric, 18 distinct): ['2', '0', '7', '3', '8', '4', '1', '9', '5', '6']
'''

CONTEXT = "Employee Attrition at IBM"
TARGET = CuratedTarget(raw_name="Attrition", new_name="Employee Attrition", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["EmployeeCount", "StandardHours", "Over18"]
FEATURES = []
