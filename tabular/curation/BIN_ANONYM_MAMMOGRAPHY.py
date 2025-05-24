from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mammography
====
Examples: 11183
====
URL: https://www.openml.org/search?type=data&id=310
====
Description: **Author**:   
  
**Source**: Unknown -   
**Please cite**:   

Mammography dataset

Past Usage:
1. Woods, K., Doss, C., Bowyer, K., Solka, J., Priebe, C.,
====
Target Variable: class (nominal, 2 distinct): ['-1', '1']
====
Features:

attr1 (numeric, 5435 distinct): ['-0.7844', '-0.1311', '-0.0239', '-0.1181', '-0.0295', '-0.1281', '-0.2217', '-0.1154', '0.0108', '-0.2061']
attr2 (numeric, 883 distinct): ['-0.4702', '-0.4525', '-0.4481', '-0.4437', '-0.4392', '-0.4348', '-0.4304', '-0.4171', '-0.426', '-0.4215']
attr3 (numeric, 160 distinct): ['-0.5916', '-0.1859', '-0.3662', '-0.2761', '-0.3211', '-0.231', '-0.4113', '-0.1408', '-0.0957', '-0.0056']
attr4 (numeric, 2800 distinct): ['-0.8596', '0.9392', '0.651', '1.0877', '1.1628', '0.497', '0.7396', '0.8012', '0.459', '0.8877']
attr5 (numeric, 1739 distinct): ['-0.3779', '0.6184', '4.2067', '2.7262', '0.3862', '0.2447', '1.2688', '0.5872', '0.5986', '13.7504']
attr6 (numeric, 550 distinct): ['-0.9457', '1.108', '1.044', '1.0105', '0.9587', '1.105', '1.1598', '1.0044', '1.0349', '0.9222']
'''

CONTEXT = "Anonymized: Mammography Results"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []