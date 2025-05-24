from tabular.datasets.manual_curation_obj import CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: albert
====
Examples: 425240
====
URL: https://www.openml.org/search?type=data&id=41147
====
Description: The goal of this challenge is to expose the research community to real world datasets of interest to 4Paradigm. All datasets are formatted in a uniform way, though the type of data might differ. The data are provided as preprocessed matrices, so that participants can focus on classification, although participants are welcome to use additional feature extraction procedures (as long as they do not violate any rule of the challenge). All problems are binary classification problems and are assessed with the normalized Area Under the ROC Curve (AUC) metric (i.e. 2*AUC-1).
                   The identity of the datasets and the type of data is concealed, though its structure is revealed. The final score in  phase 2 will be the average of rankings  on all testing datasets, a ranking will be generated from such results, and winners will be determined according to such ranking.
                   The tasks are constrained by a time budget. The Codalab platform provides computational resources shared by all participants. Each code submission will be exceuted in a compute worker with the following characteristics: 2Cores / 8G Memory / 40G SSD with Ubuntu OS. To ensure the fairness of the evaluation, when a code submission is evaluated, its execution time is limited in time.
                   http://automl.chalearn.org/data
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

V1 (numeric, 234 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V2 (numeric, 3949 distinct): ['0.0', '1.0', '-1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
V3 (numeric, 1219 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
V4 (numeric, 123 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '0.0', '6.0', '7.0', '8.0', '9.0']
V5 (numeric, 47037 distinct): ['1.0', '0.0', '2.0', '4.0', '5.0', '7.0', '8.0', '10.0', '11.0', '12.0']
V6 (numeric, 2784 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V7 (numeric, 1150 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V8 (numeric, 175 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V9 (numeric, 2493 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V10 (numeric, 11 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '0.0']
V11 (numeric, 118 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V12 (numeric, 133 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V13 (numeric, 237 distinct): ['1.0', '2.0', '3.0', '0.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V14 (nominal, 957 distinct): ['25', '543', '464', '703', '918', '473', '676', '1184', '290', '847']
V15 (nominal, 545 distinct): ['126', '68', '71', '88', '128', '177', '149', '27', '200', '290']
V16 (nominal, 170908 distinct): ['NaN', '292636', '249713', '3968', '204099', '247058', '239366', '168639', '176475', '247077']
V17 (nominal, 74500 distinct): ['NaN', '105969', '22522', '10395', '114432', '72943', '135950', '50994', '57753', '133934']
V18 (nominal, 225 distinct): ['43', '103', '88', '65', '57', '15', '258', '187', '188', '259']
V19 (nominal, 14 distinct): ['6', '14', '16', 'NaN', '5', '1', '3', '11', '9', '10']
V20 (nominal, 10235 distinct): ['1274', '3154', '9590', '5033', '2886', '4215', '5710', '9154', '11117', '6354']
V21 (nominal, 442 distinct): ['21', '200', '69', '111', '8', '176', '438', '217', '239', '322']
V22 (nominal, 3 distinct): ['3', '1', '2']
V23 (nominal, 22358 distinct): ['7291', '29724', '31167', '31020', '8555', '491', '5980', '18295', '11370', '9125']
V24 (nominal, 4566 distinct): ['2297', '4412', '1485', '2505', '2731', '3236', '3534', '2486', '3362', '1270']
V25 (nominal, 153820 distinct): ['NaN', '132825', '278863', '45050', '179112', '217046', '66488', '270188', '198367', '295588']
V26 (nominal, 3099 distinct): ['1138', '662', '886', '1035', '332', '37', '1469', '1240', '392', '1416']
V27 (nominal, 26 distinct): ['3', '18', '5', '10', '20', '1', '26', '13', '8', '21']
V28 (nominal, 7774 distinct): ['1678', '7779', '569', '593', '2034', '8315', '6247', '8969', '5872', '2048']
V29 (nominal, 121479 distinct): ['NaN', '167684', '126184', '233281', '236903', '51474', '177696', '173690', '188875', '47391']
V30 (nominal, 10 distinct): ['10', '1', '9', '5', '7', '4', '6', '2', '3', '8']
V31 (nominal, 3539 distinct): ['3707', '631', '2142', '3062', '1433', '1419', '1979', '889', '1368', '1918']
V32 (nominal, 1655 distinct): ['NaN', '224', '602', '646', '810', '1080', '195', '1495', '1702', '194']
V33 (nominal, 4 distinct): ['NaN', '4', '3', '1']
V34 (nominal, 139752 distinct): ['NaN', '128357', '102', '88786', '273713', '257814', '4665', '44342', '107330', '251390']
V35 (nominal, 11 distinct): ['NaN', '9', '11', '5', '7', '10', '12', '6', '4', '2']
V36 (nominal, 14 distinct): ['2', '3', '4', '10', '13', '11', '5', '14', '8', '12']
V37 (nominal, 27266 distinct): ['11075', '31333', '10245', '4082', 'NaN', '12092', '38828', '25245', '30534', '25035']
V38 (nominal, 61 distinct): ['NaN', '1', '56', '59', '48', '8', '39', '13', '61', '2']
V39 (nominal, 26343 distinct): ['1', 'NaN', '11981', '33211', '7766', '32232', '28199', '11980', '25129', '17599']
V40 (numeric, 1133 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V41 (nominal, 4 distinct): ['NaN', '4', '3', '1']
V42 (numeric, 170 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V43 (numeric, 2767 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V44 (nominal, 121505 distinct): ['NaN', '167684', '126184', '233281', '236903', '177696', '51474', '173690', '188875', '47391']
V45 (nominal, 10 distinct): ['10', '1', '9', '5', '7', '4', '6', '2', '3', '8']
V46 (nominal, 10230 distinct): ['1274', '3154', '9590', '5033', '2886', '4215', '5710', '9154', '6354', '11117']
V47 (nominal, 14 distinct): ['6', '14', '16', 'NaN', '5', '1', '3', '11', '9', '10']
V48 (nominal, 225 distinct): ['43', '103', '88', '65', '57', '15', '258', '187', '188', '259']
V49 (nominal, 545 distinct): ['126', '68', '71', '88', '128', '177', '149', '27', '200', '290']
V50 (numeric, 1152 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V51 (numeric, 119 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V52 (numeric, 11 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '0.0', '8.0']
V53 (numeric, 134 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V54 (nominal, 544 distinct): ['126', '68', '71', '88', '128', '177', '149', '27', '200', '290']
V55 (nominal, 170898 distinct): ['NaN', '292636', '249713', '3968', '204099', '247058', '239366', '168639', '176475', '247077']
V56 (nominal, 22371 distinct): ['7291', '29724', '31167', '31020', '8555', '491', '5980', '18295', '11370', '9125']
V57 (nominal, 26381 distinct): ['1', 'NaN', '11981', '33211', '7766', '32232', '28199', '11980', '25129', '17599']
V58 (nominal, 27182 distinct): ['11075', '31333', '10245', '4082', 'NaN', '12092', '38828', '25245', '30534', '25035']
V59 (numeric, 1162 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V60 (nominal, 546 distinct): ['126', '68', '71', '88', '128', '177', '149', '27', '200', '290']
V61 (nominal, 3098 distinct): ['1138', '662', '886', '1035', '332', '37', '1469', '1240', '392', '1416']
V62 (nominal, 3103 distinct): ['1138', '662', '886', '1035', '332', '37', '1469', '1240', '392', '1416']
V63 (nominal, 14 distinct): ['2', '3', '4', '10', '13', '11', '5', '14', '8', '12']
V64 (numeric, 137 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V65 (nominal, 26 distinct): ['3', '18', '5', '10', '20', '1', '13', '26', '8', '21']
V66 (nominal, 10227 distinct): ['1274', '9590', '3154', '5033', '2886', '4215', '5710', '9154', '6354', '11117']
V67 (numeric, 137 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V68 (nominal, 26 distinct): ['3', '18', '5', '10', '20', '1', '26', '13', '8', '21']
V69 (numeric, 133 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V70 (nominal, 62 distinct): ['NaN', '1', '56', '59', '48', '8', '39', '13', '61', '2']
V71 (nominal, 60 distinct): ['NaN', '1', '56', '59', '48', '8', '39', '13', '61', '2']
V72 (numeric, 119 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V73 (nominal, 227 distinct): ['43', '103', '88', '65', '57', '15', '258', '187', '188', '259']
V74 (nominal, 220 distinct): ['43', '103', '88', '65', '57', '15', '258', '187', '188', '259']
V75 (numeric, 118 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V76 (nominal, 229 distinct): ['43', '103', '88', '65', '57', '15', '258', '187', '188', '259']
V77 (nominal, 7786 distinct): ['1678', '7779', '569', '593', '2034', '8315', '6247', '5872', '8969', '2048']
V78 (nominal, 3103 distinct): ['1138', '662', '886', '1035', '332', '37', '1469', '1240', '392', '1416']
'''

CONTEXT = "Anonymized Dataset: Albert"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []