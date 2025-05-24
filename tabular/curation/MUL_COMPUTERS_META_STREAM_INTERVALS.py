from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: meta_stream_intervals.arff
====
Examples: 45164
====
URL: https://www.openml.org/search?type=data&id=279
====
Description: None
====
Target Variable: class (nominal, 11 distinct): ['moa.LeveragingBag_HoeffdingTree(1)', 'moa.OzaBag_HoeffdingTree(1)', 'moa.OzaBoost_HoeffdingTree(1)', 'moa.LeveragingBag_kNN(1)', 'moa.NaiveBayes(1)', 'moa.kNN(1)', 'moa.SPegasos(1)', 'moa.WEKAClassifier_SMO_PolyKernel(1)', 'moa.WEKAClassifier_OneR(1)', 'moa.WEKAClassifier_J48(1)']
====
Features:

openml_task_id (numeric, 49 distinct): ['188.0', '2150.0', '174.0', '192.0', '160.0', '190.0', '170.0', '2162.0', '191.0', '2164.0']
openml_interval_start (numeric, 1455 distinct): ['13000.0', '1000.0', '45000.0', '26000.0', '44000.0', '32000.0', '17000.0', '9000.0', '31000.0', '3000.0']
openml_interval_end (numeric, 1455 distinct): ['14000.0', '2000.0', '46000.0', '27000.0', '45000.0', '33000.0', '18000.0', '10000.0', '32000.0', '4000.0']
openml_classifier_moa.WEKAClassifier_J48(1) (numeric, 32271 distinct): ['0.8357', '0.836', '0.9197', '0.9355', '0.8357', '0.8349', '0.8359', '0.5097', '0.9004', '0.8359']
openml_classifier_moa.LeveragingBag_HoeffdingTree(1) (numeric, 36462 distinct): ['0.9853', '0.9596', '0.9851', '0.9851', '0.9596', '0.9853', '0.9851', '0.9596', '0.9853', '0.9848']
openml_classifier_moa.WEKAClassifier_SMO_PolyKernel(1) (numeric, 34333 distinct): ['0.9704', '0.9255', '0.8569', '0.76', '0.9316', '0.6544', '0.8949', '0.9254', '0.939', '0.7822']
openml_classifier_moa.kNN(1) (numeric, 28094 distinct): ['0.8481', '0.9641', '0.9636', '0.9641', '0.8221', '0.9642', '0.9642', '0.892', '0.9636', '0.9642']
openml_classifier_moa.WEKAClassifier_OneR(1) (numeric, 32658 distinct): ['0.5214', '0.173', '0.2704', '0.204', '0.7236', '0.8481', '0.2039', '0.8481', '0.8397', '0.939']
openml_classifier_moa.LeveragingBag_kNN(1) (numeric, 27780 distinct): ['0.9641', '0.892', '0.892', '0.9641', '0.8574', '0.9641', '0.8919', '0.892', '0.9641', '0.9641']
openml_classifier_moa.OzaBoost_HoeffdingTree(1) (numeric, 39140 distinct): ['0.8937', '0.9844', '0.9846', '0.9846', '0.984', '0.9514', '0.9846', '0.8835', '0.868', '0.9235']
openml_classifier_moa.HoeffdingTree(1) (numeric, 39756 distinct): ['0.939', '0.9397', '0.9398', '0.8925', '0.9389', '0.9244', '0.9244', '0.9397', '0.8722', '0.9391']
openml_classifier_moa.SGD(1) (numeric, 33477 distinct): ['0.1669', '0.1668', '0.1669', '0.1669', '0.1669', '0.1669', '0.1669', '0.1668', '0.1667', '0.1669']
openml_classifier_moa.NaiveBayes(1) (numeric, 30876 distinct): ['0.9862', '0.9863', '0.9862', '0.9862', '0.8063', '0.906', '0.9862', '0.9863', '0.9862', '0.935']
openml_classifier_moa.WEKAClassifier_REPTree(1) (numeric, 33602 distinct): ['0.8545', '0.8259', '0.6388', '0.8315', '0.8259', '0.8259', '0.9226', '0.4967', '0.6389', '0.6389']
openml_classifier_moa.OzaBag_HoeffdingTree(1) (numeric, 39043 distinct): ['0.9845', '0.9844', '0.9844', '0.9403', '0.9846', '0.9845', '0.9844', '0.888', '0.9845', '0.9403']
openml_classifier_moa.SPegasos(1) (numeric, 37902 distinct): ['0.0168', '0.0168', '0.0168', '0.0168', '0.0167', '0.0168', '0.1092', '0.0168', '0.0168', '0.0168']
meta_REPTreeDepth2ErrRate (numeric, 831 distinct): ['24.6', '23.6', '65.9', '24.5', '24.3', '65.3', '66.2', '22.5', '65.0', '22.7']
meta_J48.00001.ErrRate (numeric, 762 distinct): ['20.9', '29.9', '19.2', '30.3', '21.2', '30.1', '30.2', '29.5', '20.0', '29.8']
meta_NBErrRate (numeric, 731 distinct): ['13.2', '12.9', '12.3', '13.1', '12.6', '12.8', '13.3', '13.0', '12.5', '13.6']
meta_MeanMutualInformation (numeric, 24191 distinct): ['-1.0', '0.0038', '0.0049', '0.0043', '0.0042', '0.0043', '0.0035', '0.0036', '0.0038', '0.0038']
meta_NBAUC (numeric, 40124 distinct): ['0.9995', '0.9995', '0.9993', '0.946', '0.9993', '0.9912', '0.9983', '0.9995', '0.9878', '0.9379']
meta_DecisionStumpKappa (numeric, 38448 distinct): ['0.0', '0.0231', '0.1071', '0.114', '0.0934', '0.1051', '0.1063', '0.304', '0.0189', '0.1081']
meta_HoeffdingDDM.warnings (numeric, 31 distinct): ['0', '1', '12', '11', '2', '3', '13', '10', '7', '14']
meta_NoiseToSignalRatio (numeric, 27159 distinct): ['-1.0', '7.9033', '-14.6825', '1.0227', '14.7262', '2.2604', '7.3934', '-6.0531', '10.328', '7.1177']
meta_RandomTreeDepth3AUC_K=0 (numeric, 41781 distinct): ['0.844', '0.753', '0.7134', '0.7186', '0.7085', '0.8482', '0.75', '0.8305', '0.7161', '0.8376']
meta_PercentageOfNumericAtts (numeric, 27 distinct): ['0.0', '0.9091', '0.9412', '0.75', '0.1538', '0.3014', '0.9792', '0.9836', '0.1579', '0.375']
meta_EquivalentNumberOfAtts (numeric, 27169 distinct): ['-1.0', '15.7896', '16.9851', '7.8605', '6.4279', '-13.6825', '6.2236', '5.8334', '6.3345', '13.9347']
meta_HoeffdingDDM.changes (numeric, 20 distinct): ['0', '1', '9', '2', '6', '7', '10', '8', '4', '5']
meta_ClassEntropy (numeric, 15806 distinct): ['-1.0', '0.9996', '0.9997', '0.9993', '0.997', '0.999', '0.9988', '0.9991', '0.9969', '0.9997']
meta_NaiveBayesDdm.changes (numeric, 21 distinct): ['0', '10', '1', '8', '7', '9', '6', '11', '4', '2']
meta_NumNominalAtts (numeric, 25 distinct): ['0', '13', '32', '9', '29', '50', '7', '15', '64', '36']
meta_REPTreeDepth3AUC (numeric, 40035 distinct): ['0.5', '0.4955', '0.4958', '0.4956', '0.4953', '0.4959', '0.4995', '0.496', '0.4995', '0.4961']
meta_MeanAttributeEntropy (numeric, 15634 distinct): ['-1.0', '2.8071', '0.8745', '1.9642', '1.2632', '1.9641', '0.9643', '0.9634', '1.964', '1.9645']
meta_MeanKurtosisOfNumericAtts (numeric, 30695 distinct): ['0.0', '-1.1942', '-1.1928', '-1.1906', '-1.1974', '-1.1953', '-1.2076', '-1.1935', '-1.2001', '-1.1967']
meta_REPTreeDepth3Kappa (numeric, 41090 distinct): ['0.0', '0.556', '0.944', '0.2306', '0.3007', '0.258', '0.4024', '0.7318', '0.3092', '0.9539']
meta_J48.001.ErrRate (numeric, 753 distinct): ['16.3', '16.2', '15.9', '15.6', '19.2', '17.6', '18.5', '16.8', '15.7', '17.8']
meta_NumNumericAtts (numeric, 20 distinct): ['0', '10', '6', '3', '16', '22', '7', '47', '15', '13']
meta_ClassCount (numeric, 10 distinct): ['2', '5', '10', '6', '4', '7', '3', '19', '11', '26']
meta_J48.00001.AUC (numeric, 38779 distinct): ['0.5', '0.4958', '0.4955', '0.4956', '0.4959', '0.496', '0.4988', '0.4988', '0.4953', '0.4988']
meta_PercentageOfBinaryAtts (numeric, 23 distinct): ['0.0', '0.2143', '0.6667', '0.6027', '0.3', '0.1429', '0.4737', '0.25', '0.9189', '0.1739']
meta_DecisionStumpAUC (numeric, 40510 distinct): ['0.6277', '0.699', '0.7131', '0.7114', '0.6597', '0.7122', '0.6748', '0.7214', '0.7064', '0.7124']
meta_RandomTreeDepth1AUC_K=0 (numeric, 41486 distinct): ['0.696', '0.6988', '0.7109', '0.6582', '0.6224', '0.6883', '0.6995', '0.6923', '0.5807', '0.6722']
meta_REPTreeDepth2Kappa (numeric, 40525 distinct): ['0.0', '0.5663', '0.348', '0.944', '0.5675', '0.49', '0.0256', '0.012', '0.2825', '0.2509']
meta_PositivePercentage (numeric, 458 distinct): ['0.0', '0.001', '0.002', '0.086', '0.087', '0.088', '0.085', '0.089', '0.084', '0.083']
meta_J48.0001.ErrRate (numeric, 759 distinct): ['15.5', '19.2', '18.9', '17.3', '29.3', '18.5', '16.3', '29.2', '15.8', '30.1']
meta_MinNominalAttDistinctValues (numeric, 6 distinct): ['-1.0', '2.0', '1.0', '3.0', '4.0', '7.0']
meta_RandomTreeDepth2AUC_K=0 (numeric, 41883 distinct): ['0.8178', '0.7792', '0.6342', '0.6785', '0.8122', '0.7574', '0.641', '0.663', '0.8387', '0.6984']
meta_MeanMeansOfNumericAtts (numeric, 29817 distinct): ['0.0', '0.4994', '0.4959', '0.5001', '0.4963', '0.5035', '0.4988', '0.5007', '0.4979', '0.5014']
meta_J48.0001.kappa (numeric, 39647 distinct): ['0.0', '0.6202', '0.7677', '0.642', '0.524', '0.7097', '0.534', '0.5877', '0.571', '0.404']
meta_MeanNominalAttDistinctValues (numeric, 27 distinct): ['-1.0', '2.0', '2.2414', '2.3', '2.9333', '2.9375', '4.5556', '5.6667', '2.0556', '2.7143']
meta_J48.00001.kappa (numeric, 39520 distinct): ['0.0', '0.5722', '0.7097', '0.5628', '0.5742', '0.5706', '0.5788', '0.556', '0.5879', '0.576']
meta_PercentageOfNominalAtts (numeric, 30 distinct): ['0.0', '0.9667', '0.8205', '0.6849', '0.96', '0.7895', '0.9846', '0.5625', '0.9', '0.973']
meta_REPTreeDepth1Kappa (numeric, 37613 distinct): ['0.0', '-0.0006', '0.0987', '0.1478', '-0.0006', '-0.0006', '-0.0011', '0.3734', '-0.0011', '0.012']
meta_NegativePercentage (numeric, 774 distinct): ['0.114', '0.113', '0.115', '0.112', '0.116', '0.111', '0.117', '0.11', '0.118', '0.119']
meta_NumBinaryAtts (numeric, 15 distinct): ['0', '3', '4', '44', '2', '20', '16', '13', '1', '17']
meta_NaiveBayesDdm.warnings (numeric, 37 distinct): ['0', '1', '12', '3', '11', '13', '2', '10', '8', '5']
meta_MaxNominalAttDistinctValues (numeric, 15 distinct): ['-1.0', '3.0', '4.0', '2.0', '7.0', '8.0', '5.0', '11.0', '10.0', '9.0']
meta_J48.001.AUC (numeric, 38908 distinct): ['0.5', '0.4958', '0.4955', '0.4956', '0.496', '0.4959', '0.4953', '0.4988', '0.4988', '0.495']
meta_J48.0001.AUC (numeric, 38832 distinct): ['0.5', '0.4958', '0.4955', '0.4956', '0.4959', '0.496', '0.4953', '0.4988', '0.4988', '0.4988']
meta_NBKappa (numeric, 42417 distinct): ['0.766', '0.756', '0.778', '0.788', '0.9422', '0.77', '0.752', '0.0999', '0.1159', '0.7298']
meta_REPTreeDepth1AUC (numeric, 37613 distinct): ['0.5', '0.4958', '0.4955', '0.4956', '0.496', '0.4959', '0.4953', '0.4995', '0.495', '0.4988']
meta_NaiveBayesAdwin.changes (numeric, 12 distinct): ['0', '1', '2', '4', '3', '5', '6', '7', '8', '9']
meta_REPTreeDepth3ErrRate (numeric, 763 distinct): ['19.9', '19.3', '18.6', '19.4', '18.7', '19.2', '19.5', '18.2', '20.0', '19.0']
meta_DecisionStumpErrRate (numeric, 887 distinct): ['80.0', '79.8', '79.9', '79.7', '80.1', '80.2', '79.5', '80.3', '79.3', '79.2']
meta_MeanStdDevOfNumericAtts (numeric, 29435 distinct): ['0.0', '0.2883', '0.2874', '0.0671', '0.288', '0.2891', '0.289', '0.289', '0.2888', '0.2878']
meta_Dimensionality (numeric, 27 distinct): ['0.011', '0.014', '0.017', '0.039', '0.02', '0.065', '0.035', '0.004', '0.03', '0.019']
meta_REPTreeDepth2AUC (numeric, 39901 distinct): ['0.5', '0.4958', '0.4955', '0.4956', '0.496', '0.4953', '0.4959', '0.4995', '0.495', '0.4961']
meta_StdvNominalAttDistinctValues (numeric, 24 distinct): ['-1.0', '0.0', '0.7395', '0.9091', '7.5056', '1.6242', '0.3507', '4.1866', '4.4721', '0.2323']
meta_HoeffdingAdwin.changes (numeric, 13 distinct): ['0', '1', '2', '4', '3', '5', '6', '7', '8', '9']
meta_MeanSkewnessOfNumericAtts (numeric, 30152 distinct): ['0.0', '-0.0073', '-0.0153', '-0.0245', '0.0072', '-0.0087', '-0.0035', '0.0021', '-0.0049', '0.0025']
meta_DefaultAccuracy (numeric, 774 distinct): ['0.114', '0.113', '0.115', '0.112', '0.116', '0.111', '0.117', '0.11', '0.118', '0.119']
meta_REPTreeDepth1ErrRate (numeric, 909 distinct): ['80.0', '80.2', '80.1', '79.5', '79.8', '79.9', '79.6', '79.2', '80.3', '79.7']
meta_J48.001.kappa (numeric, 39745 distinct): ['0.0', '0.7572', '0.448', '0.7945', '0.6854', '0.398', '0.5871', '0.6016', '0.624', '0.628']
meta_NumAttributes (numeric, 27 distinct): ['11', '14', '17', '39', '20', '65', '35', '4', '30', '19']
'''

CONTEXT = "Meta Stream intervals"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []