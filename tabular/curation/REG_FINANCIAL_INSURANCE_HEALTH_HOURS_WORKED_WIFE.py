from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: health_insurance
====
Examples: 22272
====
URL: https://www.openml.org/search?type=data&id=44993
====
Description: **Data Description**

Dataset is a cross-section study from 1993 in United States.

It presents dataset about health insurance and hours worked by wives. Each instance is a data about a married woman.

**Attribute Description**

1. *whrswk* - hours worked per week by wife, target feature
2. *hhi* - whether wife covered by husband's health insurance
3. *whi* - whether wife has health insurance through her job ?
4. *hhi2* - whether husband has health insurance through her job ?
5. *education* - a factor with levels, "<9years", "9-11years", "12years", "13-15years", "16years", ">16years"
6. *race* - "white", "black", "other"
7. *hispanic* - "yes" or "no"
8". *experience* - years of potential work experience
9. *kidslt6* - number of kids under age of 6
10. *kids618* - number of kids 6-18 years old
11. *husby* - husband's income in thousands of dollars
12. *region* - one of "other", "northcentral", "south", "west"
13. *wght* - sampling weight (should be ignored)
====
Target Variable: whrswk (numeric, 75 distinct): ['40', '0', '35', '30', '20', '50', '45', '38', '25', '32']
====
Features:

hhi (nominal, 2 distinct): ['no', 'yes']
whi (nominal, 2 distinct): ['no', 'yes']
hhi2 (nominal, 2 distinct): ['yes', 'no']
education (nominal, 6 distinct): ['12years', '13-15years', '16years', '9-11years', '>16years', '<9years']
race (nominal, 3 distinct): ['white', 'black', 'other']
hispanic (nominal, 2 distinct): ['no', 'yes']
experience (numeric, 100 distinct): ['21.0', '15.0', '16.0', '20.0', '17.0', '19.0', '18.0', '14.0', '13.0', '11.0']
kidslt6 (numeric, 6 distinct): ['0', '1', '2', '3', '4', '5']
kids618 (numeric, 9 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8']
husby (numeric, 2540 distinct): ['0.0', '30.0', '40.0', '35.0', '99.999', '25.0', '50.0', '20.0', '32.0', '45.0']
region (nominal, 4 distinct): ['south', 'northcentral', 'other', 'west']
'''

CONTEXT = "Health Insurance and Hours Worked by Wife"
TARGET = CuratedTarget(raw_name="whrswk", new_name="Hours Worked per Week by Wife", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="hhi", new_name="Wife Covered by Husband's Health Insurance"),
            CuratedFeature(raw_name="whi", new_name="Wife has Health Insurance through her Job"),
            CuratedFeature(raw_name="hhi2", new_name="Husband has Health Insurance through his Job"),
            CuratedFeature(raw_name="education", new_name="Education",
                           value_mapping={'<9years': 'Less than 9 years', '9-11years': '9-11 years', '12years': '12 years',
                                          '13-15years': '13-15 years', '16years': '16 years',
                                          '>16years': 'More than 16 years'}),
            CuratedFeature(raw_name='experience', new_name='Years of Potential Work Experience'),
            CuratedFeature(raw_name='kidslt6', new_name='Number of Kids under Age of 6'),
            CuratedFeature(raw_name='kids618', new_name='Number of Kids 6-18 Years Old'),
            CuratedFeature(raw_name='husby', new_name="Husband's Income in Thousands of Dollars"),
            CuratedFeature(raw_name='region', value_mapping={'northcentral': 'North Central'})]