from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Employee-Turnover-at-TECHCO
====
Examples: 34452
====
URL: https://www.openml.org/search?type=data&id=43551
====
Description: Context
These are simulated data based on employee turnover data in a real technology company in India (we refer to this company by a pseudonym, 'TECHCO'). These data can be used to analyze drivers of turnover at TECHCO. The original dataset was analyzed in the paper Machine Learning for Pattern Discovery in Management Research (SSRN version here). This publicly offered dataset is simulated based on the original data for privacy considerations. Along with the accompanying Python Kaggle code and R Kaggle code, this dataset will help readers learn how to implement the ML techniques in the paper. The data and code demonstrate how ML can be useful for discovering nonlinear and interactive patterns between variables that may otherwise have gone unnoticed. 
Content
This dataset includes 1,191 entry-level employees that were quasi-randomly deployed to any of TECHCOs nine geographically dispersed production centers in 2007. The data are structured as a panel with one observation for each month that an individual is employed at the company for up to 40 months. The data include 34,453 observations from 1,191 employees total; The dependent variable, Turnover, indicates whether the employee left or stayed during that time period. 
Objectives
The objective in the original paper was to explore patterns in the data that would help us learn more about the drivers of employee turnover. Another objective could be to find the best predictive model to estimate when a specific employee will leave.
====
Target Variable: turnover (string, 2 distinct): ['Stayed', 'Left']
====
Features:

time (numeric, 39 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
training_score (numeric, 1190 distinct): ['4.683', '4.8404', '4.9192', '4.2164', '4.1634', '4.7748', '4.3967', '4.8491', '4.8121', '4.7556']
logical_score (numeric, 18 distinct): ['0', '1', '6', '7', '3', '4', '9', '5', '10', '8']
verbal_score (numeric, 25 distinct): ['0', '4', '3', '2', '1', '5', '6', '7', '8', '9']
avg_literacy (numeric, 1190 distinct): ['83.5885', '81.0521', '78.4636', '75.7561', '66.0504', '73.0548', '80.3122', '85.7793', '83.2417', '81.9356']
location_age (numeric, 21 distinct): ['10', '25', '11', '24', '9', '26', '12', '23', '7', '8']
distance (numeric, 1087 distinct): ['0.0', '1.6355', '1.1395', '0.0729', '0.1554', '0.0674', '2.4079', '0.8376', '0.8797', '2.0804']
similar_language (numeric, 941 distinct): ['100.0', '1.25', '24.1105', '94.7218', '30.6313', '98.6468', '21.9529', '63.925', '12.4524', '35.8779']
is_male (numeric, 2 distinct): ['1', '0']
'''

CONTEXT = "Employee Turnover at TECHCO - Indian Technology Company"
TARGET = CuratedTarget(raw_name="turnover", new_name="Employee Turnover", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
