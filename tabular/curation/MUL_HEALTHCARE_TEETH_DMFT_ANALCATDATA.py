from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: analcatdata_dmft
====
Examples: 797
====
URL: https://www.openml.org/search?type=data&id=469
====
Description: **Author**: Unknown   
**Source**: [Jeffrey S. Simonoff](http://people.stern.nyu.edu/jsimonof/AnalCatData/Data/) - 2003    
**Please cite**: Jeffrey S. Simonoff, Analyzing Categorical Data, Springer-Verlag, 2003

One of the datasets used in the book "Analyzing Categorical Data,"
by Jeffrey S. Simonoff. It contains data on the DMFT Index (Decayed, Missing, and Filled Teeth) before and after different prevention strategies. The prevention strategy is commonly used as the (categorical) target.

### Attribute information  
* DMFT.Begin and DMFT.End: DMFT index before and after the prevention strategy
* Gender of the individual
* Ethnicity of the individual
====
Target Variable: Prevention (nominal, 6 distinct): ['Mouthwash', 'None', 'Diet_enrichment', 'All_methods', 'Health_education', 'Oral_hygiene']
====
Features:

DMFT.Begin (nominal, 9 distinct): ['0', '2', '4', '6', '5', '3', '1', '7', '8']
DMFT.End (nominal, 7 distinct): ['0', '1', '2', '3', '4', '5', '6']
Gender (nominal, 2 distinct): ['Male', 'Female']
Ethnic (nominal, 3 distinct): ['White', 'Dark', 'Black']
'''

CONTEXT = "DMFT Index (Decayed, Missing, and Filled Teeth) before and after different prevention strategies"
TARGET = CuratedTarget(raw_name="Prevention", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []