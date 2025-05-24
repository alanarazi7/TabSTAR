from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: dionis
====
Examples: 416188
====
URL: https://www.openml.org/search?type=data&id=41167
====
Description: SOURCE: [ChaLearn Automatic Machine Learning Challenge (AutoML)](https://competitions.codalab.org/competitions/2321), [ChaLearn](https://automl.chalearn.org/data) 

This is a "supervised learning" challenge in machine learning. We are making available 30 datasets, all pre-formatted in given feature representations (this means that each example consists of a fixed number of numerical coefficients). The challenge is to solve classification and regression problems, without any further human intervention.

The difficulty is that there is a broad diversity of data types and distributions (including balanced or unbalanced classes, sparse or dense feature representations, with or without missing values or categorical variables, various metrics of evaluation, various proportions of number of features and number of examples). The problems are drawn from a wide variety of domains and include medical diagnosis from laboratory analyses, speech recognition, credit rating, prediction or drug toxicity or efficacy, classification of text, prediction of customer satisfaction, object recognition, protein structure prediction, action recognition in video data, etc. While there exist machine learning toolkits including methods that can solve all these problems, it is still considerable human effort to find, for a given combination of dataset, task, metric of evaluation, and available computational time, the combination of methods and hyper-parameter setting that is best suited. Your challenge is to create the "perfect black box" eliminating the human in the loop.

This is a challenge with code submission: your code will be executed automatically on our servers to train and test your learning machines with unknown datasets. However, there is NO OBLIGATION TO SUBMIT CODE. Half of the prizes can be won by just submitting prediction results. There are six rounds (Prep, Novice, Intermediate, Advanced, Expert, and Master) in which datasets of progressive difficulty are introduced (5 per round). There is NO PREREQUISITE TO PARTICIPATE IN PREVIOUS ROUNDS to enter a new round. The rounds alternate AutoML phases in which submitted code is "blind tested" in limited time on our platform, using datasets you have never seen before, and Tweakathon phases giving you time to improve your methods by tweaking them on those datasets and running them on your own systems (without computational resource limitation).

NOTE: This dataset corresponds to one of the datasets of the challenge.
====
Target Variable: class (nominal, 355 distinct): ['120', '36', '121', '318', '101', '0', '34', '82', '187', '327']
====
Features:

V1 (numeric, 4925 distinct): ['9464.0', '9402.0', '9351.0', '9452.0', '9470.0', '9173.0', '9465.0', '9333.0', '9435.0', '9477.0']
V2 (numeric, 8044 distinct): ['0.0', '415.0', '-13.0', '523.0', '72.0', '536.0', '191.0', '363.0', '31.0', '343.0']
V3 (numeric, 1309 distinct): ['10000.0', '0.0', '36276.0', '28099.0', '47496.0', '41888.0', '56834.0', '52279.0', '-9069.0', '64892.0']
V4 (numeric, 58054 distinct): ['0.0', '-117.0', '18.0', '-28.0', '-134.0', '-62.0', '-142.0', '-181.0', '-37.0', '-234.0']
V5 (numeric, 4087 distinct): ['0.0', '106.0', '-45.0', '13.0', '-7.0', '5.0', '21.0', '-21.0', '-10.0', '-43.0']
V6 (numeric, 1182 distinct): ['10000.0', '11547.0', '0.0', '11926.0', '13859.0', '15333.0', '16885.0', '18226.0', '19549.0', '20750.0']
V7 (numeric, 20836 distinct): ['0.0', '-5.0', '-21.0', '-22.0', '19.0', '-31.0', '-2.0', '6.0', '4.0', '-18.0']
V8 (numeric, 61135 distinct): ['0.0', '6.0', '15.0', '54.0', '-1.0', '-17.0', '172.0', '-159.0', '-76.0', '202.0']
V9 (numeric, 4979 distinct): ['0.0', '-152.0', '-51.0', '16.0', '29.0', '-39.0', '84.0', '105.0', '89.0', '-23.0']
V10 (numeric, 1886 distinct): ['0.0', '20.0', '22.0', '18.0', '16.0', '31.0', '54.0', '9.0', '50.0', '-6.0']
V11 (numeric, 9314 distinct): ['10000.0', '5000.0', '6667.0', '6000.0', '5833.0', '6250.0', '5625.0', '5714.0', '6111.0', '6364.0']
V12 (numeric, 34664 distinct): ['0.0', '1228.0', '788.0', '-932.0', '-162.0', '-137.0', '-45.0', '-157.0', '-118.0', '-141.0']
V13 (numeric, 16974 distinct): ['-10000.0', '-9999.0', '-9998.0', '-9997.0', '-9996.0', '-9995.0', '-9994.0', '-9993.0', '-9992.0', '-9990.0']
V14 (numeric, 1 distinct): ['0']
V15 (numeric, 58575 distinct): ['0.0', '209.0', '140.0', '154.0', '-4.0', '94.0', '105.0', '-71.0', '-740.0', '160.0']
V16 (numeric, 8624 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '9995.0', '9993.0', '9994.0', '9992.0', '7273.0']
V17 (numeric, 9123 distinct): ['10000.0', '9999.0', '9998.0', '6667.0', '9997.0', '9995.0', '9996.0', '9993.0', '6250.0', '9992.0']
V18 (numeric, 41067 distinct): ['0.0', '12.0', '18.0', '10.0', '9.0', '1.0', '-5.0', '3.0', '13.0', '-2.0']
V19 (numeric, 7747 distinct): ['0.0', '303.0', '430.0', '840.0', '369.0', '407.0', '389.0', '383.0', '405.0', '199.0']
V20 (numeric, 3815 distinct): ['0.0', '157.0', '164.0', '218.0', '147.0', '122.0', '318.0', '151.0', '153.0', '161.0']
V21 (numeric, 2967 distinct): ['0.0', '150.0', '-11.0', '78.0', '67.0', '40.0', '90.0', '18.0', '15.0', '83.0']
V22 (numeric, 7276 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '9995.0', '9993.0', '9994.0', '9990.0', '9991.0']
V23 (numeric, 82253 distinct): ['0.0', '-384.0', '-155.0', '-350.0', '-71.0', '-361.0', '-78.0', '-330.0', '-210.0', '-221.0']
V24 (numeric, 40314 distinct): ['0.0', '2.0', '8.0', '-2.0', '4.0', '13.0', '-12.0', '-6.0', '9.0', '3.0']
V25 (numeric, 1459 distinct): ['0.0', '34.0', '16.0', '17.0', '35.0', '47.0', '-20.0', '2.0', '14.0', '-4.0']
V26 (numeric, 2620 distinct): ['0.0', '-7.0', '-25.0', '54.0', '6.0', '28.0', '79.0', '10.0', '37.0', '47.0']
V27 (numeric, 1 distinct): ['0']
V28 (numeric, 11383 distinct): ['0.0', '7225.0', '7748.0', '8856.0', '9612.0', '10469.0', '4081.0', '4186.0', '3883.0', '3758.0']
V29 (numeric, 9674 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '6667.0', '9995.0', '7500.0', '9993.0', '9992.0']
V30 (numeric, 49007 distinct): ['0.0', '22.0', '-15.0', '-20.0', '44.0', '20.0', '21.0', '43.0', '29.0', '33.0']
V31 (numeric, 6888 distinct): ['0.0', '223.0', '255.0', '135.0', '146.0', '92.0', '206.0', '98.0', '-14.0', '221.0']
V32 (numeric, 7478 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '9995.0', '9994.0', '9992.0', '9993.0', '9988.0']
V33 (numeric, 1 distinct): ['0']
V34 (numeric, 8279 distinct): ['0.0', '-659.0', '-1179.0', '-362.0', '-536.0', '-322.0', '-240.0', '469.0', '204.0', '252.0']
V35 (numeric, 1 distinct): ['0']
V36 (numeric, 4925 distinct): ['536.0', '598.0', '649.0', '548.0', '530.0', '827.0', '535.0', '667.0', '565.0', '523.0']
V37 (numeric, 1 distinct): ['0']
V38 (numeric, 16869 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '9995.0', '9994.0', '9992.0', '9991.0', '9993.0']
V39 (numeric, 6229 distinct): ['0.0', '264.0', '211.0', '240.0', '93.0', '87.0', '232.0', '212.0', '167.0', '134.0']
V40 (numeric, 30659 distinct): ['0.0', '158.0', '295.0', '290.0', '111.0', '156.0', '126.0', '148.0', '68.0', '312.0']
V41 (numeric, 66076 distinct): ['0.0', '206.0', '218.0', '228.0', '172.0', '-13467.0', '211.0', '-1347.0', '192.0', '167.0']
V42 (numeric, 52856 distinct): ['0.0', '70.0', '29.0', '154.0', '57.0', '107.0', '117.0', '96.0', '104.0', '210.0']
V43 (numeric, 34775 distinct): ['0.0', '-235.0', '-169.0', '-146.0', '-162.0', '-228.0', '-126.0', '-156.0', '-242.0', '-221.0']
V44 (numeric, 4541 distinct): ['0.0', '-819.0', '-777.0', '-890.0', '-802.0', '-589.0', '-854.0', '-721.0', '-827.0', '-548.0']
V45 (numeric, 2254 distinct): ['0.0', '30.0', '43.0', '79.0', '48.0', '116.0', '74.0', '24.0', '62.0', '29.0']
V46 (numeric, 58989 distinct): ['0.0', '-46.0', '-57.0', '-72.0', '-77.0', '-81.0', '-68.0', '-44.0', '-66.0', '-91.0']
V47 (numeric, 6838 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '9995.0', '9994.0', '9993.0', '9989.0', '9988.0']
V48 (numeric, 59559 distinct): ['0.0', '2565.0', '3215.0', '3773.0', '4264.0', '-2565.0', '4706.0', '-2300.0', '-2202.0', '-2349.0']
V49 (numeric, 1694 distinct): ['0.0', '50.0', '26.0', '36.0', '59.0', '32.0', '31.0', '13.0', '51.0', '28.0']
V50 (numeric, 62447 distinct): ['0.0', '79.0', '155.0', '-260.0', '-373.0', '-50.0', '12.0', '-359.0', '-364.0', '193.0']
V51 (numeric, 8165 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '6667.0', '7500.0', '9995.0', '9993.0', '7143.0']
V52 (numeric, 4471 distinct): ['0.0', '-154.0', '-4082.0', '-113.0', '-502.0', '-77.0', '-206.0', '85.0', '48.0', '25.0']
V53 (numeric, 38 distinct): ['8', '9', '10', '11', '12', '7', '13', '14', '15', '16']
V54 (numeric, 1 distinct): ['0']
V55 (numeric, 8499 distinct): ['10000.0', '9999.0', '9998.0', '9997.0', '9996.0', '9995.0', '9994.0', '9993.0', '9992.0', '9991.0']
V56 (numeric, 63618 distinct): ['0.0', '130.0', '135.0', '-59.0', '143.0', '125.0', '-11.0', '76.0', '121.0', '-58.0']
V57 (numeric, 55797 distinct): ['0.0', '-31.0', '-17.0', '-11.0', '-36.0', '-9.0', '-16.0', '-30.0', '-24.0', '-26.0']
V58 (numeric, 5006 distinct): ['0.0', '119.0', '98.0', '103.0', '3.0', '137.0', '140.0', '96.0', '138.0', '115.0']
V59 (numeric, 22748 distinct): ['0.0', '-28.0', '-54.0', '-31.0', '-35.0', '-59.0', '-15.0', '-50.0', '-62.0', '-48.0']
V60 (numeric, 1565 distinct): ['0.0', '3.0', '13.0', '-2.0', '8.0', '7.0', '29.0', '28.0', '-5.0', '-12.0']
'''

CONTEXT = "Anonymized Dataset: Dionis"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []