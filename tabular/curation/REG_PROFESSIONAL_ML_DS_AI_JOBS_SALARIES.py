from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: REG_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES
====
Examples: 103013
====
URL: https://ai-jobs.net/salaries/download/salaries.csv
====
Description: 
ML/DS Salaries
(https://ai-jobs.net/salaries/download/salaries.csv)
salary and basic information of workers in machine learning and data science industry. The task is to predict the salary of workers.

====
Features:

work_year (int64, 6 distinct): ['2024', '2025', '2023', '2022', '2021', '2020']
experience_level (object, 4 distinct): ['SE', 'MI', 'EN', 'EX']
employment_type (object, 4 distinct): ['FT', 'PT', 'CT', 'FL']
job_title (object, 338 distinct): ['Data Scientist', 'Data Engineer', 'Software Engineer', 'Data Analyst', 'Machine Learning Engineer', 'Engineer', 'Manager', 'Research Scientist', 'Analyst', 'Applied Scientist']
salary (int64, 9318 distinct): ['160000', '110000', '150000', '100000', '180000', '200000', '120000', '140000', '130000', '90000']
salary_currency (object, 26 distinct): ['USD', 'GBP', 'EUR', 'CAD', 'INR', 'PLN', 'CHF', 'PHP', 'AUD', 'BRL']
salary_in_usd (int64, 10294 distinct): ['160000', '110000', '150000', '180000', '100000', '200000', '120000', '140000', '130000', '170000']
employee_residence (object, 98 distinct): ['US', 'CA', 'GB', 'AU', 'DE', 'FR', 'NL', 'LT', 'ES', 'AT']
remote_ratio (int64, 3 distinct): ['0', '100', '50']
company_location (object, 92 distinct): ['US', 'CA', 'GB', 'AU', 'DE', 'FR', 'NL', 'LT', 'ES', 'AT']
company_size (object, 3 distinct): ['M', 'L', 'S']
'''

CONTEXT = "Salaries of ML/DS Professionals Worldwide"
TARGET = CuratedTarget(raw_name="salary", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["salary_in_usd"]
FEATURES = []

DESCRIPTION = '''
ML/DS Salaries
(https://ai-jobs.net/salaries/download/salaries.csv)
salary and basic information of workers in machine learning and data science industry. The task is to predict the salary of workers.
'''