from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cmc
====
Examples: 1473
====
URL: https://www.openml.org/search?type=data&id=23
====
Description: **Author**: [Tjen-Sien Lim](limt@stat.wisc.edu) 
**Source**: [As obtained from UCI](https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice)
**Please cite**: [UCI citation](https://archive.ics.uci.edu/ml/citation_policy.html)

1. Title: Contraceptive Method Choice
 
 2. Sources:
    (a) Origin:  This dataset is a subset of the 1987 National Indonesia
                 Contraceptive Prevalence Survey
    (b) Creator: Tjen-Sien Lim (limt@stat.wisc.edu)
    (c) Donor:   Tjen-Sien Lim (limt@stat.wisc.edu)
    (c) Date:    June 7, 1997
 
 3. Past Usage:
    Lim, T.-S., Loh, W.-Y. & Shih, Y.-S. (1999). A Comparison of
    Prediction Accuracy, Complexity, and Training Time of Thirty-three
    Old and New Classification Algorithms. Machine Learning. Forthcoming.
    (ftp://ftp.stat.wisc.edu/pub/loh/treeprogs/quest1.7/mach1317.pdf or
    (http://www.stat.wisc.edu/~limt/mach1317.pdf)
 
 4. Relevant Information:
    This dataset is a subset of the 1987 National Indonesia Contraceptive
    Prevalence Survey. The samples are married women who were either not 
    pregnant or do not know if they were at the time of interview. The 
    problem is to predict the current contraceptive method choice 
    (no use, long-term methods, or short-term methods) of a woman based 
    on her demographic and socio-economic characteristics.
 
 5. Number of Instances: 1473
 
 6. Number of Attributes: 10 (including the class attribute)
 
 7. Attribute Information:
 
    1. Wife's age                     (numerical)
    2. Wife's education               (categorical)      1=low, 2, 3, 4=high
    3. Husband's education            (categorical)      1=low, 2, 3, 4=high
    4. Number of children ever born   (numerical)
    5. Wife's religion                (binary)           0=Non-Islam, 1=Islam
    6. Wife's now working?            (binary)           0=Yes, 1=No
    7. Husband's occupation           (categorical)      1, 2, 3, 4
    8. Standard-of-living index       (categorical)      1=low, 2, 3, 4=high
    9. Media exposure                 (binary)           0=Good, 1=Not good
    10. Contraceptive method used     (class attribute)  1=No-use 
                                                         2=Long-term
                                                         3=Short-term
 
 8. Missing Attribute Values: None

 Information about the dataset
 CLASSTYPE: nominal
 CLASSINDEX: last
====
Target Variable: Contraceptive_method_used (nominal, 3 distinct): ['1', '3', '2']
====
Features:

Wifes_age (numeric, 34 distinct): ['25', '26', '30', '32', '28', '35', '24', '29', '22', '27']
Wifes_education (nominal, 4 distinct): ['4', '3', '2', '1']
Husbands_education (nominal, 4 distinct): ['4', '3', '2', '1']
Number_of_children_ever_born (numeric, 15 distinct): ['1', '2', '3', '4', '5', '0', '6', '7', '8', '9']
Wifes_religion (nominal, 2 distinct): ['1', '0']
Wifes_now_working%3F (nominal, 2 distinct): ['1', '0']
Husbands_occupation (nominal, 4 distinct): ['3', '1', '2', '4']
Standard-of-living_index (nominal, 4 distinct): ['4', '3', '2', '1']
Media_exposure (nominal, 2 distinct): ['0', '1']
'''

QUALITY_MAPPING = {'1': 'Low', '2': 'Medium', '3': 'High', '4': 'Very High'}

CONTEXT = "Indonesian Woman Contraceptive Prevalence Survey"
TARGET = CuratedTarget(raw_name="Contraceptive_method_used", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'No Use', '2': 'Long Term', '3': 'Short Term'})
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name="Wifes_education", value_mapping=QUALITY_MAPPING),
            CuratedFeature(raw_name="Husbands_education", value_mapping=QUALITY_MAPPING),
            CuratedFeature(raw_name="Wifes_religion", value_mapping={'1': 'Islam', '0': 'Non-Islam'}),
            CuratedFeature(raw_name="Wifes_now_working%3F", new_name="Wife Working"),
            CuratedFeature(raw_name="Standard-of-living_index", value_mapping=QUALITY_MAPPING),
            CuratedFeature(raw_name="Media_exposure", value_mapping={'1': 'Not Good', '0': 'Good'})
            ]