from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: sylvine
====
Examples: 5124
====
URL: https://www.openml.org/search?type=data&id=41146
====
Description: SOURCE: [ChaLearn Automatic Machine Learning Challenge (AutoML)](https://competitions.codalab.org/competitions/2321), [ChaLearn](https://automl.chalearn.org/data) 

This is a "supervised learning" challenge in machine learning. We are making available 30 datasets, all pre-formatted in given feature representations (this means that each example consists of a fixed number of numerical coefficients). The challenge is to solve classification and regression problems, without any further human intervention.

The difficulty is that there is a broad diversity of data types and distributions (including balanced or unbalanced classes, sparse or dense feature representations, with or without missing values or categorical variables, various metrics of evaluation, various proportions of number of features and number of examples). The problems are drawn from a wide variety of domains and include medical diagnosis from laboratory analyses, speech recognition, credit rating, prediction or drug toxicity or efficacy, classification of text, prediction of customer satisfaction, object recognition, protein structure prediction, action recognition in video data, etc. While there exist machine learning toolkits including methods that can solve all these problems, it is still considerable human effort to find, for a given combination of dataset, task, metric of evaluation, and available computational time, the combination of methods and hyper-parameter setting that is best suited. Your challenge is to create the "perfect black box" eliminating the human in the loop.

This is a challenge with code submission: your code will be executed automatically on our servers to train and test your learning machines with unknown datasets. However, there is NO OBLIGATION TO SUBMIT CODE. Half of the prizes can be won by just submitting prediction results. There are six rounds (Prep, Novice, Intermediate, Advanced, Expert, and Master) in which datasets of progressive difficulty are introduced (5 per round). There is NO PREREQUISITE TO PARTICIPATE IN PREVIOUS ROUNDS to enter a new round. The rounds alternate AutoML phases in which submitted code is "blind tested" in limited time on our platform, using datasets you have never seen before, and Tweakathon phases giving you time to improve your methods by tweaking them on those datasets and running them on your own systems (without computational resource limitation).

NOTE: This dataset corresponds to one of the datasets of the challenge.
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

V1 (numeric, 157 distinct): ['221', '216', '230', '220', '232', '229', '227', '226', '228', '224']
V2 (numeric, 364 distinct): ['0.0', '13.0', '3.0', '4.0', '2.0', '6.0', '7.0', '23.0', '16.0', '8.0']
V3 (numeric, 118 distinct): ['229', '228', '233', '232', '223', '231', '236', '225', '227', '226']
V4 (numeric, 2549 distinct): ['930.0', '541.0', '1410.0', '150.0', '765.0', '1140.0', '808.0', '900.0', '1008.0', '684.0']
V5 (numeric, 360 distinct): ['45.0', '0.0', '90.0', '13.0', '81.0', '18.0', '6.0', '63.0', '108.0', '356.0']
V6 (numeric, 2574 distinct): ['997.0', '150.0', '900.0', '430.0', '1020.0', '1140.0', '607.0', '1231.0', '750.0', '1383.0']
V7 (numeric, 1183 distinct): ['3372.0', '3397.0', '3373.0', '3370.0', '3387.0', '3389.0', '3362.0', '3330.0', '3338.0', '3365.0']
V8 (numeric, 48 distinct): ['11', '13', '14', '9', '10', '12', '15', '8', '7', '16']
V9 (numeric, 2620 distinct): ['1410.0', '150.0', '511.0', '1471.0', '212.0', '2089.0', '1657.0', '2206.0', '474.0', '1530.0']
V10 (numeric, 389 distinct): ['30.0', '0.0', '150.0', '60.0', '67.0', '108.0', '42.0', '85.0', '90.0', '120.0']
V11 (numeric, 120 distinct): ['233', '237', '225', '229', '227', '221', '223', '220', '235', '218']
V12 (numeric, 151 distinct): ['228', '230', '226', '218', '222', '224', '217', '223', '231', '232']
V13 (numeric, 146 distinct): ['226', '230', '228', '224', '235', '233', '234', '237', '229', '227']
V14 (numeric, 2556 distinct): ['618.0', '1020.0', '150.0', '1087.0', '1140.0', '849.0', '1061.0', '997.0', '1050.0', '1465.0']
V15 (numeric, 2218 distinct): ['618.0', '1087.0', '1142.0', '942.0', '1530.0', '541.0', '990.0', '1448.0', '1507.0', '2190.0']
V16 (numeric, 393 distinct): ['0.0', '7.0', '4.0', '2.0', '11.0', '1.0', '9.0', '23.0', '19.0', '10.0']
V17 (numeric, 122 distinct): ['228', '229', '226', '221', '217', '231', '224', '223', '230', '236']
V18 (numeric, 360 distinct): ['45.0', '0.0', '18.0', '90.0', '135.0', '34.0', '50.0', '72.0', '63.0', '27.0']
V19 (numeric, 361 distinct): ['45.0', '135.0', '0.0', '63.0', '90.0', '29.0', '27.0', '17.0', '68.0', '54.0']
V20 (numeric, 226 distinct): ['146', '152', '142', '145', '143', '154', '147', '125', '140', '149']
'''

CONTEXT = "Anonymized Dataset: Sylvine"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []