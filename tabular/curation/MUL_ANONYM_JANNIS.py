from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: jannis
====
Examples: 83733
====
URL: https://www.openml.org/search?type=data&id=41168
====
Description: SOURCE: [ChaLearn Automatic Machine Learning Challenge (AutoML)](https://competitions.codalab.org/competitions/2321), [ChaLearn](https://automl.chalearn.org/data) 

This is a "supervised learning" challenge in machine learning. We are making available 30 datasets, all pre-formatted in given feature representations (this means that each example consists of a fixed number of numerical coefficients). The challenge is to solve classification and regression problems, without any further human intervention.

The difficulty is that there is a broad diversity of data types and distributions (including balanced or unbalanced classes, sparse or dense feature representations, with or without missing values or categorical variables, various metrics of evaluation, various proportions of number of features and number of examples). The problems are drawn from a wide variety of domains and include medical diagnosis from laboratory analyses, speech recognition, credit rating, prediction or drug toxicity or efficacy, classification of text, prediction of customer satisfaction, object recognition, protein structure prediction, action recognition in video data, etc. While there exist machine learning toolkits including methods that can solve all these problems, it is still considerable human effort to find, for a given combination of dataset, task, metric of evaluation, and available computational time, the combination of methods and hyper-parameter setting that is best suited. Your challenge is to create the "perfect black box" eliminating the human in the loop.

This is a challenge with code submission: your code will be executed automatically on our servers to train and test your learning machines with unknown datasets. However, there is NO OBLIGATION TO SUBMIT CODE. Half of the prizes can be won by just submitting prediction results. There are six rounds (Prep, Novice, Intermediate, Advanced, Expert, and Master) in which datasets of progressive difficulty are introduced (5 per round). There is NO PREREQUISITE TO PARTICIPATE IN PREVIOUS ROUNDS to enter a new round. The rounds alternate AutoML phases in which submitted code is "blind tested" in limited time on our platform, using datasets you have never seen before, and Tweakathon phases giving you time to improve your methods by tweaking them on those datasets and running them on your own systems (without computational resource limitation).

NOTE: This dataset corresponds to one of the datasets of the challenge.
====
Target Variable: class (nominal, 4 distinct): ['3', '1', '2', '0']
====
Features:

V1 (numeric, 49883 distinct): ['1.0', '0.0042', '0.0073', '0.0092', '0.0065', '0.0058', '0.0179', '0.0151', '0.0227', '0.0112']
V2 (numeric, 854 distinct): ['1.0', '0.1417', '0.1', '0.1167', '0.1667', '0.0917', '0.175', '0.1333', '0.1917', '0.1083']
V3 (numeric, 871 distinct): ['1.0', '0.15', '0.125', '0.2083', '0.1583', '0.225', '0.2', '0.25', '0.175', '0.1917']
V4 (numeric, 78429 distinct): ['0.501', '0.5014', '0.4934', '0.488', '0.5112', '0.4606', '0.5113', '0.5048', '0.5021', '0.4841']
V5 (numeric, 80119 distinct): ['0.5014', '0.501', '0.6596', '0.4321', '0.7631', '0.5871', '0.7568', '0.6152', '0.4999', '0.7552']
V6 (numeric, 77198 distinct): ['1.0', '1.0041', '1.0004', '1.0043', '1.0008', '1.0', '1.0066', '1.0043', '1.0', '1.0002']
V7 (numeric, 79001 distinct): ['1.0', '0.1342', '0.2603', '0.154', '0.2977', '0.2858', '0.3987', '0.1361', '0.3329', '0.1356']
V8 (numeric, 78272 distinct): ['0.0097', '0.1', '0.125', '0.1429', '0.1111', '0.0714', '0.0435', '0.0667', '0.0741', '0.1667']
V9 (numeric, 76727 distinct): ['0.0', '0.3333', '0.3', '0.25', '0.5', '0.2', '0.4', '0.2917', '0.2333', '0.3125']
V10 (numeric, 72445 distinct): ['135.452', '123.603', '126.481', '125.876', '254.0', '116.874', '112.464', '113.061', '121.621', '124.582']
V11 (numeric, 73939 distinct): ['108.573', '109.297', '109.793', '104.784', '115.143', '135.879', '106.909', '100.959', '118.731', '152.922']
V12 (numeric, 76580 distinct): ['127.201', '102.216', '113.118', '113.984', '102.024', '101.781', '113.785', '117.245', '108.612', '252.969']
V13 (numeric, 78893 distinct): ['32.0936', '22.8218', '28.3225', '38.0229', '23.823', '30.4141', '44.3853', '59.3776', '45.399', '30.6308']
V14 (numeric, 78948 distinct): ['24.0032', '30.1229', '46.764', '31.3437', '37.6425', '34.4676', '33.8713', '35.4559', '15.9714', '16.7848']
V15 (numeric, 79167 distinct): ['36.6287', '24.5051', '33.0007', '28.543', '32.4558', '39.2535', '17.0003', '36.8919', '30.2682', '33.708']
V16 (numeric, 81910 distinct): ['0.0', '1.0578', '1.0923', '1.1331', '1.6349', '1.8227', '-1.4118', '1.3479', '1.119', '0.6287']
V17 (numeric, 81865 distinct): ['0.0', '1.9416', '1.0975', '1.0899', '1.4317', '1.0045', '1.2168', '1.0148', '1.4584', '1.1449']
V18 (numeric, 81469 distinct): ['0.0', '1.4973', '1.2295', '1.6709', '1.079', '1.8108', '1.323', '1.1884', '1.5416', '1.5464']
V19 (numeric, 77497 distinct): ['74.5645', '77.6466', '70.3506', '66.6222', '77.5358', '99.6184', '77.5503', '66.0606', '66.5921', '68.7257']
V20 (numeric, 82216 distinct): ['-1.1519', '1.7819', '-1.3291', '-2.8178', '1.1748', '1.0244', '3.8927', '1.1642', '2.8309', '1.9786']
V21 (numeric, 81534 distinct): ['12.1171', '15.5758', '14.2958', '11.2294', '23.3899', '21.3598', '17.412', '11.5108', '15.0193', '2.5657']
V22 (numeric, 75989 distinct): ['12.6635', '16.1406', '11.4131', '17.4952', '12.3737', '15.3185', '16.0569', '14.1392', '13.8406', '11.2348']
V23 (numeric, 80108 distinct): ['1.2886', '1.7434', '1.3565', '1.7238', '1.0136', '1.1722', '1.0028', '2.9461', '1.0306', '1.4452']
V24 (numeric, 80124 distinct): ['10.4762', '12.3328', '10.5901', '4.191', '13.5587', '5.3979', '2.1035', '6.3322', '0.0', '1.0033']
V25 (numeric, 82018 distinct): ['-1.204', '0.4847', '-1.2175', '-0.4651', '0.2345', '0.4753', '-1.2185', '-1.0983', '-1.0557', '-1.7175']
V26 (numeric, 81783 distinct): ['2.1619', '1.1245', '1.8086', '-0.2306', '1.0661', '1.206', '0.2168', '1.486', '0.1096', '1.8542']
V27 (numeric, 81574 distinct): ['1.015', '1.1048', '1.3527', '1.0154', '0.9124', '1.0304', '1.0814', '-1.0626', '1.0586', '1.6102']
V28 (numeric, 77513 distinct): ['66.0606', '99.6184', '74.5645', '57.8917', '71.3247', '66.5921', '62.2233', '67.7996', '77.6466', '75.2947']
V29 (numeric, 857 distinct): ['1.0', '0.1417', '0.1', '0.1167', '0.1667', '0.1917', '0.0917', '0.175', '0.1333', '0.1583']
V30 (numeric, 82218 distinct): ['-1.5096', '-2.2465', '6.9099', '2.5299', '-2.1299', '2.878', '1.5274', '2.8349', '-1.3291', '1.2813']
V31 (numeric, 79105 distinct): ['46.7749', '33.0007', '25.4389', '35.7233', '43.3605', '47.0124', '40.5034', '12.8548', '27.983', '16.2168']
V32 (numeric, 78979 distinct): ['1.0', '0.2858', '0.2346', '0.1478', '0.2977', '0.2603', '0.1805', '0.1104', '0.1342', '0.146']
V33 (numeric, 75973 distinct): ['11.4131', '16.1406', '17.4952', '12.6635', '13.332', '16.0467', '12.8338', '14.9715', '15.0', '10.2536']
V34 (numeric, 78980 distinct): ['1.0', '0.1104', '0.1557', '0.1342', '0.154', '0.2603', '0.1478', '0.2899', '0.1998', '1.0466']
V35 (numeric, 82248 distinct): ['-1.1519', '1.9786', '2.878', '-2.8178', '1.5274', '10.0971', '11.5705', '1.8233', '2.5299', '1.7819']
V36 (numeric, 81452 distinct): ['0.0', '1.4973', '1.824', '1.0905', '1.7728', '1.5933', '1.182', '1.289', '1.0935', '1.2018']
V37 (numeric, 72571 distinct): ['121.621', '123.603', '110.569', '132.137', '153.85', '136.942', '126.984', '117.656', '139.238', '116.456']
V38 (numeric, 869 distinct): ['1.0', '0.15', '0.125', '0.2', '0.225', '0.2083', '0.1583', '0.1333', '0.275', '0.175']
V39 (numeric, 72535 distinct): ['121.621', '135.452', '131.24', '110.569', '137.644', '117.07', '147.282', '118.999', '116.874', '125.876']
V40 (numeric, 79167 distinct): ['38.868', '22.2035', '32.4558', '35.7233', '46.7749', '37.0808', '46.5765', '14.3181', '10.7962', '14.6967']
V41 (numeric, 81830 distinct): ['1.486', '1.5478', '1.1245', '1.8086', '1.0661', '1.0911', '1.8542', '2.1619', '1.1495', '1.4712']
V42 (numeric, 79012 distinct): ['37.5502', '65.1877', '24.0032', '46.419', '35.4529', '58.4859', '19.1831', '39.5431', '52.4927', '26.8377']
V43 (numeric, 76026 distinct): ['14.9715', '15.3185', '12.3737', '11.4131', '15.9416', '17.2053', '10.9081', '11.9256', '16.2064', '10.8822']
V44 (numeric, 73851 distinct): ['103.422', '251.997', '115.143', '104.784', '118.731', '109.793', '102.225', '108.573', '152.922', '103.844']
V45 (numeric, 78963 distinct): ['65.1877', '37.1858', '31.3437', '26.6514', '30.1229', '15.098', '17.7913', '19.1831', '35.4559', '37.1669']
V46 (numeric, 78410 distinct): ['0.501', '0.5014', '0.4934', '0.443', '0.5157', '0.4975', '0.5165', '0.4884', '0.4749', '0.5031']
V47 (numeric, 76748 distinct): ['0.0', '0.3333', '0.25', '0.3', '0.4', '0.2917', '0.5', '0.2', '0.3125', '0.1857']
V48 (numeric, 78323 distinct): ['0.0097', '0.1', '0.125', '0.1429', '0.0714', '0.1111', '0.0435', '0.1667', '0.0667', '0.0769']
V49 (numeric, 72534 distinct): ['121.621', '135.452', '254.0', '102.306', '123.061', '112.464', '108.688', '107.022', '117.07', '126.481']
V50 (numeric, 81484 distinct): ['0.0', '1.3933', '1.2103', '1.6709', '1.0732', '1.1942', '1.559', '1.1252', '1.0748', '1.2647']
V51 (numeric, 79239 distinct): ['50.4536', '46.7749', '33.0007', '32.4558', '26.4449', '28.543', '22.2035', '25.4389', '35.7233', '38.868']
V52 (numeric, 76003 distinct): ['16.1406', '16.1118', '12.0252', '12.8743', '14.9715', '12.9543', '11.7469', '11.9256', '14.1085', '13.1688']
V53 (numeric, 50002 distinct): ['1.0', '0.0042', '0.0227', '0.0092', '0.0065', '0.0126', '0.0142', '0.0119', '0.013', '0.0144']
V54 (numeric, 77225 distinct): ['1.0', '1.0041', '1.0', '1.0008', '1.0066', '1.0008', '1.0043', '1.0003', '1.02', '1.003']
'''

CONTEXT = "Anonymized Dataset: Jannis"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []