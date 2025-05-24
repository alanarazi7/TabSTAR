from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: socmob
====
Examples: 1156
====
URL: https://www.openml.org/search?type=data&id=541
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

17x17x2x2 tables of counts in GLIM-ready format used for the analyses
in Biblarz, Timothy J., and Adrian E. Raftery. 1993. "The Effects of
Family Disruption on Social Mobility." American Sociological Review
(In press). For further details of the data, see this reference.
Column 1 is father's occupation, coded as follows:
17. Professional, Self-Employed
16. Professional-Salaried
15. Manager
14. Salesman-Nonretail
13. Proprietor
12. Clerk
11. Salesman-Retail
10. Craftsman-Manufacturing
9. Craftsmen-Other
8. Craftsman-Construction
7. Service Worker
6. Operative-Nonmanufacturing
5. Operative-Manufacturing
4. Laborer-Manufacturing
3. Laborer-Nonmanufacturing
2. Farmer/Farm Manager
1. Farm Laborer
Column 2 is son's occupation, coded in the same way as father's.
Column 3 is family structure, coded 1=intact family background and
2=nonintact family background.
Column 4 is race, coded 1=white and 2=black.
Column 5 is counts for son's first occupation.
Column 6 is counts for son's current occupation.
The counts have been weighted to take account of the survey
design, which is why they are not integers.
************************************************************
***********************************************************
This file was constructed from publicly available data collected
by David Featherman and Robert Hauser in 1973: the "Occupational
Change in a Generation II" (OCG II) Survey. Permission is hereby given to
use the above data for non-commercial scholarly and teaching purposes.
If these data are used in a published article or book,
the authors, the original data (in the form given in
Biblarz and Raftery (1993), cited above), and StatLib should
all be acknowledged.


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific

https://www.openml.org/search?type=data&status=active&id=44987&sort=runs

Data Description

This dataset described social mobility, i.e. how the sons' occupations are related to their fathers' jobs.

An instance represent the number of sons that have a certain job A given the father has the job B (additionally conditioned on race and family structure).

The dataset was originally collected for the survey of "Occupational Change in a Generation II"

Attribute Description

fathers_occupation
sons_occupation
family_structure - "intact" or "nonintact"
race - "black" or "white"
counts_for_sons_first_occupation
counts_for_sons_current_occupation - target feature

====
Target Variable: counts_for_sons_current_occupation (numeric, 361 distinct): ['0.0', '0.5', '0.8', '0.9', '0.4', '0.6', '0.7', '1.6', '1.0', '1.4']
====
Features:

fathers_occupation (nominal, 17 distinct): ['Clerk', 'Operative-Manufacturing', 'Salesman-Retail', 'Salesman-Nonretail', 'Proprietor', 'Professional_Self-Employed', 'Professional-Salaried', 'Operative-Nonmanufacturing', 'Manager', 'Craftsman-Construction']
sons_occupation (nominal, 17 distinct): ['Clerk', 'Operative-Manufacturing', 'Salesman-Retail', 'Salesman-Nonretail', 'Proprietor', 'Professional_Self-Employed', 'Professional-Salaried', 'Operative-Nonmanufacturing', 'Manager', 'Craftsman-Construction']
family_structure (nominal, 2 distinct): ['intact', 'nonintact']
race (nominal, 2 distinct): ['black', 'white']
counts_for_sons_first_occupation (numeric, 358 distinct): ['0.0', '0.5', '0.8', '0.6', '0.4', '0.9', '0.7', '1.0', '1.4', '1.5']
'''

CONTEXT = "Social Mobility Occupation between generations"
TARGET = CuratedTarget(raw_name="counts_for_sons_current_occupation", new_name="Sons Current Occupation Count",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="family_structure", value_mapping={'nonintact': 'Non Intact'}),
            CuratedFeature(raw_name="counts_for_sons_first_occupation", new_name="Sons First Occupation Count")]