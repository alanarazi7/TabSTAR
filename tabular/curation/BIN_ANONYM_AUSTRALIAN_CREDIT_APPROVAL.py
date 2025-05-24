from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Australian
====
Examples: 690
====
URL: https://www.openml.org/search?type=data&id=40981
====
Description: **Author**: Confidential. Donated by Ross Quinlan  
**Source**: [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), [UCI](https://archive.ics.uci.edu/ml/datasets/Statlog+(Australian+Credit+Approval)) - 1987    
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html) 

**Important note:** This dataset is derived from [credit-approval](https://www.openml.org/d/29), even though both datasets exist individually on UCI. In this version, missing values were filled in (not clear how) and a duplicate feature was removed.  

**Australian Credit Approval**. This is the famous Australian Credit Approval dataset, originating from the StatLog project. It concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect the confidentiality of the data. 

This dataset was retrieved 2014-11-14 from the UCI site and converted to the ARFF format.

__Major changes w.r.t. version 3: dataset from UCI that matches description and data types__


### Feature information

There are 6 numerical and 8 categorical attributes, all normalized to [-1,1]. The original formatting was as follows:  

A1: 0,1 CATEGORICAL (formerly: a,b)  
A2: continuous.  
A3: continuous.  
A4: 1,2,3 CATEGORICAL (formerly: p,g,gg)  
A5: 1, 2,3,4,5, 6,7,8,9,10,11,12,13,14 CATEGORICAL (formerly: ff,d,i,k,j,aa,m,c,w, e, q, r,cc, x)  
A6: 1, 2,3, 4,5,6,7,8,9 CATEGORICAL (formerly: ff,dd,j,bb,v,n,o,h,z)  
A7: continuous.  
A8: 1, 0 CATEGORICAL (formerly: t, f)  
A9: 1, 0 CATEGORICAL (formerly: t, f)  
A10: continuous.  
A11: 1, 0 CATEGORICAL (formerly t, f)  
A12: 1, 2, 3 CATEGORICAL (formerly: s, g, p)  
A13: continuous.  
A14: continuous.  
A15: 1,2 class attribute (formerly: +,-)  

### Relevant Papers

Ross Quinlan. "Simplifying decision trees", Int J Man-Machine Studies 27, Dec 1987, pp. 221-234. 

Ross Quinlan. "C4.5: Programs for Machine Learning", Morgan Kaufmann, Oct 1992
====
Target Variable: A15 (nominal, 2 distinct): ['0', '1']
====
Features:

A1 (nominal, 2 distinct): ['1', '0']
A2 (numeric, 350 distinct): ['163.0', '72.0', '47.0', '90.0', '95.0', '32.0', '70.0', '49.0', '35.0', '83.0']
A3 (numeric, 215 distinct): ['46', '1', '73', '66', '40', '26', '17', '102', '52', '87']
A4 (nominal, 3 distinct): ['2', '1', '3']
A5 (nominal, 14 distinct): ['8', '11', '9', '3', '6', '1', '4', '13', '7', '14']
A6 (nominal, 8 distinct): ['4', '8', '5', '1', '3', '9', '2', '7']
A7 (numeric, 132 distinct): ['1', '7', '2', '26', '4', '14', '3', '37', '5', '58']
A8 (nominal, 2 distinct): ['1', '0']
A9 (nominal, 2 distinct): ['0', '1']
A10 (numeric, 23 distinct): ['1', '2', '3', '4', '7', '12', '6', '8', '5', '10']
A11 (nominal, 2 distinct): ['0', '1']
A12 (nominal, 3 distinct): ['2', '1', '3']
A13 (numeric, 171 distinct): ['1', '69', '38', '53', '32', '24', '98', '63', '45', '86']
A14 (numeric, 240 distinct): ['1', '2', '115', '150', '3', '94', '7', '6', '50', '77']
'''

CONTEXT = "Australian Credit Approval"
TARGET = CuratedTarget(raw_name="A15", new_name="Credit Approved", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []