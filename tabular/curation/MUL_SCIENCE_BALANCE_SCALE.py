from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: balance-scale
====
Examples: 625
====
URL: https://www.openml.org/search?type=data&id=11
====
Description: **Author**: Siegler, R. S. (donated by Tim Hume)  
**Source**: [UCI](http://archive.ics.uci.edu/ml/datasets/balance+scale) - 1994  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)   

**Balance Scale Weight & Distance Database**  
This data set was generated to model psychological experimental results.  Each example is classified as having the balance scale tip to the right, tip to the left, or be balanced. The attributes are the left weight, the left distance, the right weight, and the right distance. The correct way to find the class is the greater of (left-distance * left-weight) and (right-distance * right-weight). If they are equal, it is balanced.

### Attribute description  
The attributes are the left weight, the left distance, the right weight, and the right distance.

### Relevant papers  
Shultz, T., Mareschal, D., & Schmidt, W. (1994). Modeling Cognitive Development on Balance Scale Phenomena. Machine Learning, Vol. 16, pp. 59-88.
====
Target Variable: class (nominal, 3 distinct): ['L', 'R', 'B']
====
Features:

left-weight (numeric, 5 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0']
left-distance (numeric, 5 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0']
right-weight (numeric, 5 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0']
right-distance (numeric, 5 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0']
'''

CONTEXT = "Balance Scale Psychological Experiment"
TARGET = CuratedTarget(raw_name="class", new_name="Balance", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"L": "Left", "R": "Right", "B": "Balanced"})
COLS_TO_DROP = []
FEATURES = []