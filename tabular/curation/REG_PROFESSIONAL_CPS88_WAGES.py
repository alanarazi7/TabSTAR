from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cps88wages
====
Examples: 28155
====
URL: https://www.openml.org/search?type=data&id=44984
====
Description: **Data Description**

This study uses data on males from the 1988 March CPS to sample the data. The March CPS contains information on previous year's wages, schooling, industry, and occupation. We select a sample of men ages 18 to 70 with positive annual income greater than 50 Dollars in 1992, who are not self-employed nor working without pay. The wage data is deflated by the deflator of Personal Consumption Expenditure for 1992. The data contains 28,155 observations and has variables characterizing the individuals.

The goal is to estimate the wage using information about working individuals.

**Attribute Description**

1. *wage* - target feature
2. *education* - years of schooling
3. *experience* - years of potential work experience
4. *ethnicity* - race ("cauc", "afam")
5. *smsa* - whether living in SMSA ("no", "yes")
6. *region* - living region ("northeast", "midwest", "south", "west")
7. *parttime* - whether working parttime ("no", "yes")
====
Target Variable: wage (numeric, 5970 distinct): ['712.25', '474.83', '593.54', '830.96', '949.67', '617.28', '522.32', '427.35', '356.13', '569.8']
====
Features:

education (numeric, 19 distinct): ['12', '16', '14', '18', '13', '15', '11', '10', '17', '8']
experience (numeric, 67 distinct): ['13.0', '11.0', '9.0', '12.0', '10.0', '7.0', '15.0', '8.0', '6.0', '14.0']
ethnicity (nominal, 2 distinct): ['cauc', 'afam']
smsa (nominal, 2 distinct): ['yes', 'no']
region (nominal, 4 distinct): ['south', 'midwest', 'northeast', 'west']
parttime (nominal, 2 distinct): ['no', 'yes']
'''

CONTEXT = "CPS 1998 Wages"
TARGET = CuratedTarget(raw_name="wage", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []