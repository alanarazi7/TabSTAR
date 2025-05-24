from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: monks-problems-2
====
Examples: 601
====
URL: https://www.openml.org/search?type=data&id=334
====
Description: **Author**: Sebastian Thrun (Carnegie Mellon University)  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems) - October 1992  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)   

**The Monk's Problems: Problem 2**  
Once upon a time, in July 1991, the monks of Corsendonk Priory were faced with a school held in their priory, namely the 2nd European Summer School on Machine Learning. After listening more than one week to a wide variety of learning algorithms, they felt rather confused: Which algorithm would be optimal? And which one to avoid? As a consequence of this dilemma, they created a simple task on which all learning algorithms ought to be compared: the three MONK's problems.

The target concept associated with the 2nd Monk's problem is the binary outcome of the logical formula:  
MONK-2: EXACTLY TWO of {a1 = 1, a2 = 1, a3 = 1, a4 = 1, a5 = 1, a6 = 1}

In this dataset, the original train and test sets were merged to allow other sampling procedures. However, the original train-test splits can be found as one of the OpenML tasks. 

### Attribute information: 
* attr1: 1, 2, 3 
* attr2: 1, 2, 3 
* attr3: 1, 2 
* attr4: 1, 2, 3 
* attr5: 1, 2, 3, 4 
* attr6: 1, 2 

### Relevant papers  
The MONK's Problems - A Performance Comparison of Different Learning Algorithms, by S.B. Thrun, J. Bala, E. Bloedorn, I. Bratko, B. Cestnik, J. Cheng, K. De Jong, S. Dzeroski, S.E. Fahlman, D. Fisher, R. Hamann, K. Kaufman, S. Keller, I. Kononenko, J. Kreuziger, R.S. Michalski, T. Mitchell, P. Pachowicz, Y. Reich H. Vafaie, W. Van de Welde, W. Wenzel, J. Wnek, and J. Zhang. Technical Report CS-CMU-91-197, Carnegie Mellon University, Dec. 1991.
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

attr1 (nominal, 3 distinct): ['1', '2', '3']
attr2 (nominal, 3 distinct): ['2', '1', '3']
attr3 (nominal, 2 distinct): ['2', '1']
attr4 (nominal, 3 distinct): ['3', '1', '2']
attr5 (nominal, 4 distinct): ['3', '1', '2', '4']
attr6 (nominal, 2 distinct): ['2', '1']
'''

CONTEXT = "Anonymized: MONK's Problems: Problem 2"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []