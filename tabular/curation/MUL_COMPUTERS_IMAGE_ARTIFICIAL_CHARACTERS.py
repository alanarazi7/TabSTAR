from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: artificial-characters
====
Examples: 10218
====
URL: https://www.openml.org/search?type=data&id=1459
====
Description: **Author**: H. Altay Guvenir, Burak Acar, Haldun Muderrisoglu    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Artificial+Characters) - 1992  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  

This database has been artificially generated. It describes the structure of the capital letters A, C, D, E, F, G, H, L, P, R, indicated by a number 1-10, in that order (A=1,C=2,...). Each letter's structure is described by a set of segments (lines) which resemble the way an automatic program would segment an image. The dataset consists of 600 such descriptions per letter. 

Originally, each 'instance' (letter) was stored in a separate file, each consisting of between 1 and 7 segments, numbered 0,1,2,3,... Here they are merged. That means that the first 5 instances describe the first 5 segments of the first segmentation of the first letter (A). Also, the training set (100 examples) and test set (the rest) are merged. The next 7 instances describe another segmentation (also of the letter A) and so on.

### Attribute Information  

* V1: object number, the number of the segment (0,1,2,..,7)  
* V2-V5: the initial and final coordinates of a segment in a cartesian plane (XX1,YY1,XX2,YY2).  
* V6: size, this is the length of a segment computed by using the geometric distance between two points A(X1,Y1) and B(X2,Y2).
* V7: diagonal, this is the length of the diagonal of the smallest rectangle which includes the picture of the character. The value of this attribute is the same in each object.

### Relevant Papers  

M. Botta, A. Giordana, L. Saitta: "Learning Fuzzy Concept Definitions", IEEE-Fuzzy Conference, 1993.  
M. Botta, A. Giordana: "Learning Quantitative Feature in a Symbolic Environment", LNAI 542, 1991, pp. 296-305.
====
Target Variable: Class (nominal, 10 distinct): ['3', '8', '1', '2', '5', '6', '9', '4', '7', '10']
====
Features:

V1 (numeric, 8 distinct): ['0', '1', '2', '3', '4', '5', '6', '7']
V2 (numeric, 45 distinct): ['0', '2', '1', '14', '10', '17', '12', '7', '19', '16']
V3 (numeric, 63 distinct): ['0', '8', '24', '6', '12', '18', '20', '10', '14', '7']
V4 (numeric, 48 distinct): ['0', '2', '1', '14', '10', '17', '12', '19', '8', '9']
V5 (numeric, 66 distinct): ['0.0', '24.0', '6.0', '8.0', '12.0', '18.0', '10.0', '7.0', '14.0', '20.0']
V6 (numeric, 333 distinct): ['7.0', '9.0', '13.0', '11.0', '6.0', '12.0', '8.0', '20.0', '10.0', '15.0']
V7 (numeric, 511 distinct): ['31.3', '40.25', '26.83', '30.46', '34.71', '36.4', '35.78', '39.7', '42.49', '33.54']
'''

CONTEXT = "Artificial Characters Image Recognition"
TARGET = CuratedTarget(raw_name="Class", new_name="Capital Letter", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"1": "A",
                                      "2": "C",
                                      "3": "D",
                                      "4": "E",
                                      "5": "F",
                                      "6": "G",
                                      "7": "H",
                                      "8": "L",
                                      "9": "P",
                                      "10": "R"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Segment Number"),
            CuratedFeature(raw_name="V2", new_name="Coordinates XX1"),
            CuratedFeature(raw_name="V3", new_name="Coordinates YY1"),
            CuratedFeature(raw_name="V4", new_name="Coordinates XX2"),
            CuratedFeature(raw_name="V5", new_name="Coordinates YY2"),
            CuratedFeature(raw_name="V6", new_name="Size - Geometric Distance"),
            CuratedFeature(raw_name="V7", new_name="Diagonal Length of Smallest Rectangle")]