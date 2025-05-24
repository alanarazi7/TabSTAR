from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: autoUniv-au1-1000
====
Examples: 1000
====
URL: https://www.openml.org/search?type=data&id=1547
====
Description: **Author**: Ray. J. Hickey   
**Source**: UCI  
**Please cite**:   

* Dataset Title:  

AutoUniv Dataset  
data problem: autoUniv-au1-1000   

* Abstract:   

AutoUniv is an advanced data generator for classifications tasks. The aim is to reflect the nuances and heterogeneity of real data. Data can be generated in .csv, ARFF or C4.5 formats.

* Source:  

AutoUniv was developed by Ray. J. Hickey. Email: ray.j.hickey '@' gmail.com 
AutoUniv web-site: http://sites.google.com/site/autouniv/.


* Data Set Information:

The user first creates a classification model and then generates classified examples from it. To create a model, the following are specified: the number of attributes (up to 1000) and their type (discrete or continuous), the number of classes (up to 10), the complexity of the underlying rules and the noise level. AutoUniv then produces a model through a process of constrained randomised search to satisfy the user's requirements. A model can have up to 3000 rules. Rare class models can be designed. A sequence of models can be designed to reflect concept and/or population drift. 

AutoUniv creates three text files for a model: a Prolog specification of the model used to generate examples (.aupl); a user-friendly statement of the classification rules in an 'if ... then' format (.aurules); a statistical summary of the main properties of the model, including its Bayes rate (.auprops).


* Attribute Information: 

Attributes may be discrete with up to 10 values or continuous. A discrete attribute can be nominal with values v1, v2, v3 ... or integer with values 0, 1, 2 , ... .


* Relevant Papers:

Marrs, G, Hickey, RJ and Black, MM (2010) Modeling the example life-cycle in an online classification learner. In Proceedings of HaCDAIS 2010: International Workshop on Handling Concept Drift in Adaptive Information Systems. 
[Web Link]#proc . 

Marrs, G, Hickey, RJ and Black, MM (2010) The Impact of Latency on Online Classification Learning with Concept Drift. In Y. Bi and M.A. Williams (Eds.): KSEM 2010, LNAI 6291, Springer-Verlag, Berlin, pp. 459â€“469. 

Hickey, RJ (2007) Structure and Majority Classes in Decision Tree Learning. Journal of Machine Learning Research, 8, pp. 1747-1768.
====
Target Variable: Class (nominal, 2 distinct): ['class2', 'class1']
====
Features:

V1 (numeric, 2 distinct): ['1', '0']
V2 (numeric, 2 distinct): ['1', '0']
V3 (numeric, 2 distinct): ['1', '0']
V4 (numeric, 2 distinct): ['0', '1']
V5 (numeric, 2 distinct): ['0', '1']
V6 (numeric, 2 distinct): ['0', '1']
V7 (numeric, 2 distinct): ['1', '0']
V8 (numeric, 2 distinct): ['0', '1']
V9 (numeric, 2 distinct): ['0', '1']
V10 (numeric, 2 distinct): ['1', '0']
V11 (numeric, 2 distinct): ['0', '1']
V12 (numeric, 2 distinct): ['1', '0']
V13 (numeric, 2 distinct): ['1', '0']
V14 (numeric, 2 distinct): ['1', '0']
V15 (numeric, 2 distinct): ['1', '0']
V16 (numeric, 2 distinct): ['0', '1']
V17 (numeric, 2 distinct): ['0', '1']
V18 (numeric, 2 distinct): ['1', '0']
V19 (numeric, 2 distinct): ['0', '1']
V20 (numeric, 2 distinct): ['1', '0']
'''

CONTEXT = "Anonymized Dataset: AutoUniv-au1-1000 advanced data generator for classifications tasks"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []