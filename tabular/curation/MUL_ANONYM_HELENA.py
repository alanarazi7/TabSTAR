from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: helena
====
Examples: 65196
====
URL: https://www.openml.org/search?type=data&id=41169
====
Description: **SOURCE:** [ChaLearn Automatic Machine Learning Challenge (AutoML)](https://competitions.codalab.org/competitions/2321), [ChaLearn](https://automl.chalearn.org/data)
This is a "supervised learning" challenge in machine learning. We are making available 30 datasets, all pre-formatted in given feature representations (this means that each example consists of a fixed number of numerical coefficients). The challenge is to solve classification and regression problems, without any further human intervention.

The difficulty is that there is a broad diversity of data types and distributions (including balanced or unbalanced classes, sparse or dense feature representations, with or without missing values or categorical variables, various metrics of evaluation, various proportions of number of features and number of examples). The problems are drawn from a wide variety of domains and include medical diagnosis from laboratory analyses, speech recognition, credit rating, prediction or drug toxicity or efficacy, classification of text, prediction of customer satisfaction, object recognition, protein structure prediction, action recognition in video data, etc. While there exist machine learning toolkits including methods that can solve all these problems, it is still considerable human effort to find, for a given combination of dataset, task, metric of evaluation, and available computational time, the combination of methods and hyper-parameter setting that is best suited. Your challenge is to create the "perfect black box" eliminating the human in the loop.

This is a challenge with code submission: your code will be executed automatically on our servers to train and test your learning machines with unknown datasets. However, there is NO OBLIGATION TO SUBMIT CODE. Half of the prizes can be won by just submitting prediction results. There are six rounds (Prep, Novice, Intermediate, Advanced, Expert, and Master) in which datasets of progressive difficulty are introduced (5 per round). There is NO PREREQUISITE TO PARTICIPATE IN PREVIOUS ROUNDS to enter a new round. The rounds alternate AutoML phases in which submitted code is "blind tested" in limited time on our platform, using datasets you have never seen before, and Tweakathon phases giving you time to improve your methods by tweaking them on those datasets and running them on your own systems (without computational resource limitation).

**NOTE:** This dataset corresponds to one of the datasets of the challenge.
====
Target Variable: class (nominal, 100 distinct): ['78', '55', '40', '39', '38', '17', '69', '91', '88', '77']
====
Features:

V1 (numeric, 42419 distinct): ['1.0', '0.0065', '0.013', '0.0092', '0.0089', '0.0061', '0.0226', '0.0062', '0.004', '0.0063']
V2 (numeric, 846 distinct): ['1.0', '0.1417', '0.1667', '0.1', '0.0917', '0.1167', '0.1833', '0.1083', '0.15', '0.1917']
V3 (numeric, 866 distinct): ['1.0', '0.15', '0.2', '0.275', '0.2083', '0.125', '0.225', '0.25', '0.1333', '0.1583']
V4 (numeric, 62001 distinct): ['0.501', '0.5014', '0.4934', '0.5048', '0.5141', '0.5081', '0.4666', '0.5021', '0.4817', '0.5165']
V5 (numeric, 63025 distinct): ['0.5014', '0.501', '0.2209', '0.7123', '0.1083', '0.3549', '0.8558', '0.6641', '0.7131', '0.5973']
V6 (numeric, 61014 distinct): ['1.0', '1.0165', '1.0029', '1.0004', '1.0008', '1.0003', '1.0004', '1.0026', '1.0012', '1.0642']
V7 (numeric, 62359 distinct): ['1.0', '0.1557', '0.1342', '0.2858', '0.1805', '0.2852', '0.2091', '0.2196', '0.3172', '0.4151']
V8 (numeric, 61798 distinct): ['0.0097', '0.1', '0.125', '0.1429', '0.0714', '0.1111', '0.1538', '0.069', '0.0769', '0.0909']
V9 (numeric, 60808 distinct): ['0.0', '0.3333', '0.25', '0.3', '0.4', '0.5', '0.2', '0.1667', '0.4444', '0.2917']
V10 (numeric, 58200 distinct): ['121.621', '113.061', '117.07', '147.282', '135.452', '121.081', '126.481', '115.407', '113.228', '118.973']
V11 (numeric, 58985 distinct): ['115.143', '117.488', '109.793', '104.784', '252.001', '100.13', '127.874', '111.319', '107.046', '123.743']
V12 (numeric, 60800 distinct): ['105.108', '252.967', '124.163', '105.886', '115.327', '114.235', '115.826', '113.118', '118.364', '147.495']
V13 (numeric, 62209 distinct): ['32.0936', '22.8218', '30.4141', '33.7933', '11.2816', '44.6119', '39.1892', '30.5512', '17.0264', '30.3442']
V14 (numeric, 62292 distinct): ['30.1229', '24.0032', '19.9555', '37.1858', '19.9729', '39.5431', '53.8304', '21.6613', '41.541', '32.928']
V15 (numeric, 62394 distinct): ['38.868', '50.4536', '24.5051', '38.0835', '18.1751', '17.3551', '44.5147', '29.3575', '35.7233', '29.2886']
V16 (numeric, 64062 distinct): ['0.0', '1.0333', '1.0129', '1.1092', '1.7371', '-0.1478', '1.1538', '1.0885', '-1.3056', '1.2362']
V17 (numeric, 64104 distinct): ['0.0', '1.0899', '1.154', '1.1334', '1.0479', '1.0346', '1.1255', '1.1086', '1.0032', '-1.2666']
V18 (numeric, 63824 distinct): ['0.0', '1.1433', '1.7322', '1.2552', '-1.325', '1.1783', '1.2211', '-1.3546', '1.3898', '1.6957']
V19 (numeric, 61370 distinct): ['66.0606', '59.848', '57.8917', '71.3297', '65.2992', '74.7567', '98.4492', '99.5947', '60.9991', '74.3544']
V20 (numeric, 64274 distinct): ['3.525', '3.8332', '-2.2465', '1.0244', '1.1513', '2.878', '2.8083', '1.1642', '-1.5096', '-1.4009']
V21 (numeric, 63866 distinct): ['12.1171', '12.6735', '16.19', '27.8483', '10.4533', '12.9865', '12.2013', '17.9304', '11.8648', '12.439']
V22 (numeric, 60414 distinct): ['11.4131', '11.6182', '15.2541', '10.8039', '12.7041', '13.274', '10.5214', '10.6276', '14.2143', '15.8504']
V23 (numeric, 62948 distinct): ['3.1614', '1.5408', '2.2161', '3.0084', '2.3657', '4.3034', '3.7341', '1.0332', '1.4646', '2.3754']
V24 (numeric, 63014 distinct): ['10.8327', '11.0207', '5.4396', '12.0263', '13.0081', '3.8862', '10.0454', '11.1562', '12.219', '10.0397']
V25 (numeric, 64164 distinct): ['-1.1456', '-1.2102', '-1.6782', '-0.5865', '-1.5104', '-1.0485', '-1.5312', '-1.0683', '-1.1984', '-1.1303']
V26 (numeric, 64054 distinct): ['1.9207', '1.1495', '1.3416', '-1.4409', '1.0148', '-0.6389', '1.206', '1.0989', '1.0132', '-0.3394']
V27 (numeric, 63929 distinct): ['1.3527', '1.6102', '1.1966', '1.0171', '1.0154', '1.1513', '0.4313', '1.1949', '1.4832', '1.2977']
'''

CONTEXT = "Anonymized Dataset: Helena"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []