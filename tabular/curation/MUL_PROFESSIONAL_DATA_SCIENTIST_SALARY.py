from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: data_scientist_salary
====
Examples: 15841
====
URL: https://www.openml.org/search?type=data&id=46664
====
Description: salary: Predict the salary range listed in data scientist job postings (in India) given the job description
    as well as other features like skill requirements and location. Intuitively, the best models will learn to
    identify valuable requirements from the text and high salary locations (via categorical modeling)
    as well as predictive interaction-effects. Representing a task with many text fields, this dataset
    originally stems from a 2018 MachineHack prediction competition: https://machinehack.com/hackathons/predict_the_data_scientists_salary_in_india_hackathon/overview
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: salary (string, 6 distinct): ['10to15', '15to25', '6to10', '0to3', '3to6', '25to50']
====
Features:

experience (string, 128 distinct): ['5-10 yrs', '2-5 yrs', '3-8 yrs', '2-7 yrs', '3-5 yrs', '4-9 yrs', '3-6 yrs', '7-12 yrs', '1-3 yrs', '5-8 yrs']
job_description (string, 7860 distinct): ['Accenture Technology powers our clients businesses with innovative technologies established and emerging ...', '- Experience in Credit card/ banking domain with knowledge across customer lifecycle is must;- Candidate ...', '- Experience in defining and executing professional software engineering best practices for the full ...', '- An advanced degree in Math, Computer Science, Statistics, Physics, or a related field (high GPAs ...', '- Team management / mentor ship experience is must; Should be good at resolving conflicts;- Experience ...', '- Good team management, project management and communication (both written and verbal) skills, including ...', '- Post-Graduate degree in statistics, finance, mathematics, engineering (Computer Science preferred) or ...', 'Utilize strong analytical ability to evaluate end-to-end customer experience across multiple channels ...', 'Experience leading teams of size 5-15 members;Very good knowledge of statistical techniques such as ...', '- Experience in banking domain with knowledge across customer lifecycle is must;- Candidate should have ...']
job_desig (string, 10097 distinct): ['Business Analyst', 'Data Scientist', 'Data Analyst', 'Digital Marketing Manager', 'Home Base Job/ Data Entry/online Work/part Time Work/freelancer work', 'Product Manager', 'Digital Marketing Executive', 'Analyst', 'SEO Executive', 'SEO Analyst']
job_type (string, 6 distinct): ['Analytics', 'analytics', 'ANALYTICS', 'analytic', 'Analytic']
key_skills (string, 11156 distinct): ['part time, freelancing, data entry, present job, work from home...', 'SAS, Sdtm, Adam, Statistical Programming, Statistics, Life Sciences...', 'Ar Calling, ar analyst, accounts receivable, revenue cycle management...', 'SAS, Logistic Regression, Chaid, R, Data Analytics, Anova, Excel...', 'Fraud Analytics, People Management Skills, Team Leading, Problem Solving...', 'Communication Skills, Analytical, Problem Solving, itil solving...', 'data entry operation, typing, excel, notepad, freelancing, content writing,...', 'Analytics, SAS, banking, insurance, Analytics Head', 'Excel, SQL, Data Analysis, Segmentation, SAS, Data Mining, SPSS...', 'Linear Regression, Insurance Analytics, Business Analysis...']
location (string, 1355 distinct): ['Bengaluru', 'Mumbai', 'Gurgaon', 'Pune', 'Hyderabad', 'Chennai', 'Delhi NCR', 'Noida', 'Delhi NCR, Gurgaon', 'Delhi']
'''

CONTEXT = "Indian Data Scientist Salary Prediction"
TARGET = CuratedTarget(raw_name="salary", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'10to15': '10-15',
                                      '15to25': '15-25',
                                      '6to10': '6-10',
                                      '0to3': '0-3',
                                      '3to6': '3-6', '25to50': '25-50'})
COLS_TO_DROP = []
FEATURES = []
