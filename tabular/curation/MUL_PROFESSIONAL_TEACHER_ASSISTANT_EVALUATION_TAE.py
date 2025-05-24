from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: tae
====
Examples: 151
====
URL: https://www.openml.org/search?type=data&id=48
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

1. Title: Teaching Assistant Evaluation
 
 2. Sources:
    (a) Collector: Wei-Yin Loh (Department of Statistics, UW-Madison)
    (b) Donor:     Tjen-Sien Lim (limt@stat.wisc.edu)
    (b) Date:      June 7, 1997
 
 3. Past Usage:
    1. Loh, W.-Y. & Shih, Y.-S. (1997). Split Selection Methods for 
       Classification Trees, Statistica Sinica 7: 815-840.
    2. Lim, T.-S., Loh, W.-Y. & Shih, Y.-S. (1999). A Comparison of
       Prediction Accuracy, Complexity, and Training Time of
       Thirty-three Old and New Classification Algorithms. Machine
       Learning. Forthcoming.
       (ftp://ftp.stat.wisc.edu/pub/loh/treeprogs/quest1.7/mach1317.pdf or
       (http://www.stat.wisc.edu/~limt/mach1317.pdf)
 
 4. Relevant Information:
    The data consist of evaluations of teaching performance over three
    regular semesters and two summer semesters of 151 teaching assistant
    (TA) assignments at the Statistics Department of the University of
    Wisconsin-Madison. The scores were divided into 3 roughly equal-sized
    categories ("low", "medium", and "high") to form the class variable.
 
 5. Number of Instances: 151
 
 6. Number of Attributes: 6 (including the class attribute)
 
 7. Attribute Information:
   
    1. Whether of not the TA is a native English speaker (binary)
       1=English speaker, 2=non-English speaker
    2. Course instructor (categorical, 25 categories)
    3. Course (categorical, 26 categories)
    4. Summer or regular semester (binary) 1=Summer, 2=Regular
    5. Class size (numerical)
    6. Class attribute (categorical) 1=Low, 2=Medium, 3=High
 
 8. Missing Attribute Values: None

 Information about the dataset
 CLASSTYPE: nominal
 CLASSINDEX: last
====
Target Variable: Class_attribute (nominal, 3 distinct): ['3', '2', '1']
====
Features:

Whether_of_not_the_TA_is_a_native_English_speaker (nominal, 2 distinct): ['2', '1']
Course_instructor (numeric, 25 distinct): ['23', '13', '22', '7', '9', '10', '18', '6', '15', '14']
Course (numeric, 26 distinct): ['3', '2', '1', '15', '17', '11', '7', '5', '8', '25']
Summer_or_regular_semester (nominal, 2 distinct): ['2', '1']
Class_size (numeric, 46 distinct): ['19', '20', '42', '27', '17', '38', '37', '31', '29', '10']
'''

CONTEXT = "Teacher Assistant Evaluation"
TARGET = CuratedTarget(raw_name="Class_attribute", new_name="Teacher Score", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'3': 'High', '2': 'Medium', '1': 'Low'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Whether_of_not_the_TA_is_a_native_English_speaker",
                           new_name="Native English Speaker", value_mapping={'2': 'No', '1': 'Yes'}),
            CuratedFeature(raw_name="Course_instructor", feat_type=FeatureType.CATEGORICAL),
            CuratedFeature(raw_name="Course", feat_type=FeatureType.CATEGORICAL),
            CuratedFeature(raw_name="Summer_or_regular_semester", new_name="Semester",
                           value_mapping={'2': 'Regular', '1': 'Summer'}),]