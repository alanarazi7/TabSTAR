from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER
====
Examples: 39479
====
URL: https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv
====
Description: 
https://opendata.vancouver.ca/explore/dataset/employee-remuneration-and-expenses-earning-over-75000/information/?disjunctive.department&disjunctive.title

Remuneration and expenses for employees earning over $75,000 per year. The task is to predict the remuneration of employees.

This dataset includes remuneration and expenses from employees earning over $75,000/year.

Note
Amounts are in Canadian dollars.

Asterisk (*) next to the employee name in Year 2012 and 2013 indicates exempt employee who received optional lump sum gratuity and vacation payouts. Amounts are included in the remuneration.

Data currency
The data on this site is scheduled to be updated annually.

Data accuracy
Some expenses may not reconcile within the same reporting period.

The published Statement of financial information remains the authoritative source.

Based on information recorded in the source system as at December 31st for each reporting year and does not include changes during the year.

Websites for further information
Financial reports and information
Dataset Identifier
employee-remuneration-and-expenses-earning-over-75000
Downloads
11,675
Themes
Government and finance
License
Open Government Licence - Vancouver
Modified
April 17, 2024 7:40 PM
Publisher
City of Vancouver
Data Owner
City of Vancouver
Data Team
Finance, Risk and Supply Chain Management - Accounting Operations
Search Terms
salary, salaries, staff, discretionary
Change Log
https://opendata.vancouver.ca/explore/dataset/open-data-change-log/log/?disjunctive.datasets&amp;sort=logdate&amp;refine.datasetids=employee-remuneration-and-expenses-earning-over-75000

====
Features:

year (int64, 16 distinct): ['2023', '2022', '2021', '2020', '2019', '2018', '2017', '2015', '2016', '2014']
name (object, 7228 distinct): ['Wong, B', 'Ng, W', 'Lee, D', 'Robinson, D', 'Lee, C', 'Lee, J', 'Baker, M', 'Brown, L', 'Lee, M', 'Davies, B']
department (object, 30 distinct): ['Engineering Services', 'Fire and Rescue Services', 'VFRS & OEM', 'Board of Parks & Recreation', 'IT, Digital Strategy & 311', 'Community Services', 'VFRS', 'Dev Svcs, Bldg & Licensing', 'Real Estate & Facilities Mgmt', 'Finance, Risk&Supply Chain Mgt']
title (object, 2179 distinct): ['Firefighter', 'Fire Lieutenant', 'Fire Captain', 'Superintendent I', 'Trades - Electrician', 'Journeyman - Mechanic', 'Civil Engineer I', 'Civil Engineer Ii', 'District Building Inspector', 'Planner Ii']
remuneration (float64, 34802 distinct): ['126456.0', '114675.28', '110903.0', '110103.0', '97104.0', '87968.0', '124237.0', '102840.0', '97732.0', '83851.99']
expenses (float64, 6186 distinct): ['0.0', '399.0', '998.0', '650.0', '473.0', '3086.0', '997.0', '55.0', '105.0', '546.0']
'''

CONTEXT = "Employee Remuneration and Expenses - Vancouver"
TARGET = CuratedTarget(raw_name="remuneration", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []

DESCRIPTION = '''
https://opendata.vancouver.ca/explore/dataset/employee-remuneration-and-expenses-earning-over-75000/information/?disjunctive.department&disjunctive.title

Remuneration and expenses for employees earning over $75,000 per year. The task is to predict the remuneration of employees.

This dataset includes remuneration and expenses from employees earning over $75,000/year.

Note
Amounts are in Canadian dollars.

Asterisk (*) next to the employee name in Year 2012 and 2013 indicates exempt employee who received optional lump sum gratuity and vacation payouts. Amounts are included in the remuneration.

Data currency
The data on this site is scheduled to be updated annually.

Data accuracy
Some expenses may not reconcile within the same reporting period.

The published Statement of financial information remains the authoritative source.

Based on information recorded in the source system as at December 31st for each reporting year and does not include changes during the year.

Websites for further information
Financial reports and information
Dataset Identifier
employee-remuneration-and-expenses-earning-over-75000
Downloads
11,675
Themes
Government and finance
License
Open Government Licence - Vancouver
Modified
April 17, 2024 7:40 PM
Publisher
City of Vancouver
Data Owner
City of Vancouver
Data Team
Finance, Risk and Supply Chain Management - Accounting Operations
Search Terms
salary, salaries, staff, discretionary
Change Log
https://opendata.vancouver.ca/explore/dataset/open-data-change-log/log/?disjunctive.datasets&amp;sort=logdate&amp;refine.datasetids=employee-remuneration-and-expenses-earning-over-75000
'''