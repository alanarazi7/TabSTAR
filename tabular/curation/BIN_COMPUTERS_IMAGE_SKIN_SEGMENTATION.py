from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: skin-segmentation
====
Examples: 245057
====
URL: https://www.openml.org/search?type=data&id=1502
====
Description: **Author**: Rajen Bhatt, Abhinav Dhall  
**Source**: UCI   
**Please cite**: Rajen Bhatt, Abhinav Dhall, 'Skin Segmentation Dataset', UCI Machine Learning Repository   

* Title:

Skin Segmentation Data Set 

* Abstract: 

The Skin Segmentation dataset is constructed over B, G, R color space. Skin and Nonskin dataset is generated using skin textures from face images of diversity of age, gender, and race people.

* Source:

Rajen Bhatt, Abhinav Dhall, rajen.bhatt '@' gmail.com, IIT Delhi.

* Data Set Information:

The skin dataset is collected by randomly sampling B,G,R values from face images of various age groups (young, middle, and old), race groups (white, black, and asian), and genders obtained from FERET database and PAL database. Total learning sample size is 245057; out of which 50859 is the skin samples and 194198 is non-skin samples. Color FERET Image Database: [Web Link], PAL Face Database from Productive Aging Laboratory, The University of Texas at Dallas: [Web Link]. 


* Attribute Information:

This dataset is of the dimension 245057 * 4 where first three columns are B,G,R (x1,x2, and x3 features) values and fourth column is of the class labels (decision variable y).


* Relevant Papers:

1. Rajen B. Bhatt, Gaurav Sharma, Abhinav Dhall, Santanu Chaudhury, â€œEfficient skin region segmentation using low complexity fuzzy decision tree modelâ€, IEEE-INDICON 2009, Dec 16-18, Ahmedabad, India, pp. 1-4. 
2. Abhinav Dhall, Gaurav Sharma, Rajen Bhatt, Ghulam Mohiuddin Khan, â€œAdaptive Digital Makeupâ€, in Proc. of International Symposium on Visual Computing (ISVC) 2009, Nov. 30 â€“ Dec. 02, Las Vegas, Nevada, USA, Lecture Notes in Computer Science, Vol. 5876, pp. 728-736.
====
Target Variable: Class (nominal, 2 distinct): ['2', '1']
====
Features:

V1 (numeric, 256 distinct): ['178', '179', '180', '199', '172', '181', '0', '182', '164', '173']
V2 (numeric, 256 distinct): ['178', '179', '163', '177', '175', '176', '162', '197', '172', '198']
V3 (numeric, 256 distinct): ['0', '255', '162', '131', '132', '129', '135', '128', '114', '22']
'''

CONTEXT = "Skin Image Segmentation"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="B Color Space"),
            CuratedFeature(raw_name="V2", new_name="G Color Space"),
            CuratedFeature(raw_name="V3", new_name="R Color Space")]