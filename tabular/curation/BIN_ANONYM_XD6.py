from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: xd6
====
Examples: 973
====
URL: https://www.openml.org/search?type=data&id=40693
====
Description: **Author**: Unknown  
**Source**: [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks/tree/master/datasets/classification) - Supposedly originates from UCI, but can't find it there anymore.  
**Please cite:**  

**XD6 Dataset**
Dataset used by Buntine and Niblett (1992). Composed of 10 features, one of which is irrelevant. The target is a disjunctive normal form formula over the nine other attributes, with additional classification noise.

[More info](https://books.google.be/books?id=W2bmBwAAQBAJ&pg=PA313&lpg=PA313&dq=dataset+xd6&source=bl&ots=6hYPdz8_Nl&sig=TR1ieOg9D1pCrvNyeKbb-3eKmd8&hl=en&sa=X&ved=0ahUKEwj_tZ_MxozZAhVHa1AKHZVEBBsQ6AEIQjAF#v=onepage&q=dataset xd6&f=false).
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

Attribute_1 (nominal, 2 distinct): ['0', '1']
Attribute_2 (nominal, 2 distinct): ['1', '0']
Attribute_3 (nominal, 2 distinct): ['1', '0']
Attribute_4 (nominal, 2 distinct): ['1', '0']
Attribute_5 (nominal, 2 distinct): ['0', '1']
Attribute_6 (nominal, 2 distinct): ['1', '0']
Attribute_7 (nominal, 2 distinct): ['1', '0']
Attribute_8 (nominal, 2 distinct): ['0', '1']
Attribute_9 (nominal, 2 distinct): ['1', '0']
'''

CONTEXT = "Anonymized: Disjunctive Normal Form - XD6"
TARGET = CuratedTarget(raw_name="class", new_name="Disjunctive Normal Form",
                       task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []