from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: porto-seguro
====
Examples: 595212
====
URL: https://www.openml.org/search?type=data&id=42742
====
Description: Training dataset of the 'Porto Seguros Safe Driver Prediction' Kaggle challenge [https://www.kaggle.com/c/porto-seguro-safe-driver-prediction]. The goal was to predict whether a driver will file an insurance claim next year. The official rules of the challenge explicitely state that the data may be used for 'academic research and education, and other non-commercial purposes' [https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/rules]. For a description of all variables checkout the Kaggle dataset repository [https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data]. It states that numeric features with integer values that do not contain 'bin' or 'cat' in their variable names are in fact ordinal features which could be treated as ordinal factors in R. For further information on effective preprocessing and feature engineering checkout the 'Kernels' section of the Kaggle challenge website [https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/kernels]. Note that many Kagglers removed all 'calc' variables as they do not seem to carry much information.

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.
====
Target Variable: target (nominal, 2 distinct): ['0', '1']
====
Features:

ps_ind_01 (numeric, 8 distinct): ['0', '1', '2', '5', '3', '4', '6', '7']
ps_ind_02_cat (nominal, 5 distinct): ['1', '2', '3', '4']
ps_ind_03 (numeric, 12 distinct): ['2', '3', '1', '4', '5', '6', '7', '8', '9', '0']
ps_ind_04_cat (nominal, 3 distinct): ['0', '1']
ps_ind_05_cat (nominal, 8 distinct): ['0', '6', '4', '1', '3', '2', '5']
ps_ind_06_bin (nominal, 2 distinct): ['0', '1']
ps_ind_07_bin (nominal, 2 distinct): ['0', '1']
ps_ind_08_bin (nominal, 2 distinct): ['0', '1']
ps_ind_09_bin (nominal, 2 distinct): ['0', '1']
ps_ind_10_bin (nominal, 2 distinct): ['0', '1']
ps_ind_11_bin (nominal, 2 distinct): ['0', '1']
ps_ind_12_bin (nominal, 2 distinct): ['0', '1']
ps_ind_13_bin (nominal, 2 distinct): ['0', '1']
ps_ind_14 (numeric, 5 distinct): ['0', '1', '2', '3', '4']
ps_ind_15 (numeric, 14 distinct): ['7', '8', '6', '10', '11', '9', '12', '5', '4', '13']
ps_ind_16_bin (nominal, 2 distinct): ['1', '0']
ps_ind_17_bin (nominal, 2 distinct): ['0', '1']
ps_ind_18_bin (nominal, 2 distinct): ['0', '1']
ps_reg_01 (numeric, 10 distinct): ['0.9', '0.7', '0.8', '0.6', '0.4', '0.1', '0.3', '0.5', '0.2', '0.0']
ps_reg_02 (numeric, 19 distinct): ['0.2', '0.3', '0.0', '0.4', '0.5', '0.6', '0.1', '0.7', '0.8', '0.9']
ps_reg_03 (numeric, 5013 distinct): ['0.6339', '0.6026', '0.7049', '0.5879', '0.6801', '0.5963', '0.622', '0.6685', '0.6647', '0.6315']
ps_car_01_cat (nominal, 13 distinct): ['11', '7', '6', '10', '4', '9', '5', '8', '3', '0']
ps_car_02_cat (nominal, 3 distinct): ['1', '0']
ps_car_03_cat (nominal, 3 distinct): ['1', '0']
ps_car_04_cat (nominal, 10 distinct): ['0', '1', '2', '8', '9', '6', '3', '5', '4', '7']
ps_car_05_cat (nominal, 3 distinct): ['1', '0']
ps_car_06_cat (nominal, 18 distinct): ['11', '1', '0', '14', '10', '4', '15', '6', '9', '7']
ps_car_07_cat (nominal, 3 distinct): ['1', '0']
ps_car_08_cat (nominal, 2 distinct): ['1', '0']
ps_car_09_cat (nominal, 6 distinct): ['2', '0', '1', '3', '4']
ps_car_10_cat (nominal, 3 distinct): ['1', '0', '2']
ps_car_11_cat (nominal, 104 distinct): ['104', '103', '64', '87', '32', '28', '5', '99', '65', '82']
ps_car_11 (numeric, 5 distinct): ['3.0', '2.0', '1.0', '0.0']
ps_car_12 (numeric, 184 distinct): ['0.3162', '0.4', '0.3742', '0.4472', '0.4243', '0.3161', '0.3873', '0.4899', '0.5477', '0.3997']
ps_car_13 (numeric, 70482 distinct): ['0.6746', '0.7417', '0.6928', '0.8418', '0.7492', '0.6596', '0.7478', '0.5964', '0.7437', '0.6878']
ps_car_14 (numeric, 850 distinct): ['0.3615', '0.3583', '0.3619', '0.3688', '0.3975', '0.3685', '0.3599', '0.4', '0.3937', '0.2944']
ps_car_15 (numeric, 15 distinct): ['3.6056', '3.4641', '3.3166', '3.1623', '3.7417', '2.8284', '3.0', '2.6458', '2.4495', '2.2361']
ps_calc_01 (numeric, 10 distinct): ['0.6', '0.0', '0.8', '0.5', '0.7', '0.2', '0.1', '0.3', '0.4', '0.9']
ps_calc_02 (numeric, 10 distinct): ['0.5', '0.4', '0.0', '0.3', '0.7', '0.6', '0.2', '0.1', '0.9', '0.8']
ps_calc_03 (numeric, 10 distinct): ['0.1', '0.5', '0.3', '0.6', '0.8', '0.9', '0.7', '0.0', '0.4', '0.2']
ps_calc_04 (numeric, 6 distinct): ['2', '3', '1', '4', '0', '5']
ps_calc_05 (numeric, 7 distinct): ['2', '1', '3', '0', '4', '5', '6']
ps_calc_06 (numeric, 11 distinct): ['8', '7', '9', '6', '10', '5', '4', '3', '2', '1']
ps_calc_07 (numeric, 10 distinct): ['3', '2', '4', '1', '5', '6', '0', '7', '8', '9']
ps_calc_08 (numeric, 11 distinct): ['9', '10', '8', '11', '7', '12', '6', '5', '4', '3']
ps_calc_09 (numeric, 8 distinct): ['2', '3', '1', '4', '0', '5', '6', '7']
ps_calc_10 (numeric, 26 distinct): ['8', '7', '9', '10', '6', '11', '5', '12', '4', '13']
ps_calc_11 (numeric, 20 distinct): ['5', '4', '6', '7', '3', '8', '2', '9', '10', '1']
ps_calc_12 (numeric, 11 distinct): ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
ps_calc_13 (numeric, 14 distinct): ['2', '3', '1', '4', '5', '0', '6', '7', '8', '9']
ps_calc_14 (numeric, 24 distinct): ['7', '8', '6', '9', '5', '10', '4', '11', '3', '12']
ps_calc_15_bin (nominal, 2 distinct): ['0', '1']
ps_calc_16_bin (nominal, 2 distinct): ['1', '0']
ps_calc_17_bin (nominal, 2 distinct): ['1', '0']
ps_calc_18_bin (nominal, 2 distinct): ['0', '1']
ps_calc_19_bin (nominal, 2 distinct): ['0', '1']
ps_calc_20_bin (nominal, 2 distinct): ['0', '1']
'''

CONTEXT = "Porto Seguros Safe Driver Prediction"
TARGET = CuratedTarget(raw_name="target", new_name="Claim Was Filed", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []