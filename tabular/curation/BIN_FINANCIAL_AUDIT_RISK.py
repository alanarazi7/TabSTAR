from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: audit-data
====
Examples: 1552
====
URL: https://www.openml.org/search?type=data&id=42931
====
Description: **Author**: Nishtha Hooda, CSED, TIET, Patiala

**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Audit+Data) - 2018

**Please cite**: [Hooda, Nishtha, Seema Bawa, and Prashant Singh Rana. 'Fraudulent Firm Classification: A Case Study of an External Audit.' Applied Artificial Intelligence 32.1 (2018): 48-64.]( https://doi.org/10.1080/08839514.2018.1451032)

The goal of the research is to help the auditors by building a classification model that can predict the fraudulent firm on the basis the present and historical risk factors. The information about the sectors and the counts of firms are listed respectively as Irrigation (114), Public Health (77), Buildings and Roads (82), Forest (70), Corporate (47), Animal Husbandry (95), Communication (1), Electrical (4), Land (5), Science and Technology (3), Tourism (1), Fisheries (41), Industries (37), Agriculture (200). The original dataset was separated into a trial and audit dataset. In this dataset these are concatenated into 1 dataset. Two features (trial and audit) have been added to indicate whether the data was originally from the trial or audit dataset.
====
Target Variable: Risk (numeric, 2 distinct): ['1', '0']
====
Features:

Sector_score (numeric, 13 distinct): ['55.57', '3.89', '1.85', '2.72', '3.41', '2.37', '1.99', '21.61', '59.85', '2.34']
LOCATION_ID (string, 45 distinct): ['8', '19', '9', '16', '12', '5', '2', '4', '15', '13']
PARA_A (numeric, 363 distinct): ['0.0', '0.51', '0.49', '0.56', '0.84', '1.07', '0.02', '0.29', '0.01', '0.74']
Score_A (numeric, 4 distinct): ['0.2', '0.6', '0.4']
Risk_A (numeric, 364 distinct): ['0.0', '0.102', '0.098', '0.112', '0.168', '0.428', '0.004', '0.058', '0.002', '0.148']
PARA_B (numeric, 358 distinct): ['0.0', '0.11', '0.28', '0.05', '0.46', '0.08', '0.63', '1.1', '0.25', '0.27']
Score_B (numeric, 4 distinct): ['0.2', '0.6', '0.4']
Risk_B (numeric, 361 distinct): ['0.0', '0.056', '0.022', '0.01', '0.126', '0.016', '0.092', '0.006', '0.098', '0.078']
TOTAL (numeric, 471 distinct): ['0.0', '1.1', '0.84', '0.68', '0.03', '0.07', '0.89', '0.49', '1.19', '0.21']
numbers (numeric, 5 distinct): ['5.0', '5.5', '6.0', '6.5', '9.0']
Score_B.1 (numeric, 4 distinct): ['0.2', '0.4', '0.6']
Risk_C (numeric, 6 distinct): ['1.0', '2.2', '3.6', '3.9', '5.4']
Money_Value (numeric, 329 distinct): ['0.0', '0.04', '0.02', '0.06', '0.05', '0.11', '0.01', '0.1', '0.19', '0.03']
Score_MV (numeric, 4 distinct): ['0.2', '0.6', '0.4']
Risk_D (numeric, 329 distinct): ['0.0', '0.008', '0.004', '0.012', '0.01', '0.022', '0.002', '0.02', '0.038', '0.006']
District_Loss (numeric, 4 distinct): ['2.0', '6.0', '4.0']
PROB (numeric, 4 distinct): ['0.2', '0.4', '0.6']
RiSk_E (numeric, 6 distinct): ['0.4', '1.2', '0.8', '2.4', '1.6']
History (numeric, 7 distinct): ['0', '1', '2', '3', '4', '9', '5']
Prob (numeric, 4 distinct): ['0.2', '0.4', '0.6']
Risk_F (numeric, 8 distinct): ['0.0', '0.4', '1.2', '1.8', '2.4', '5.4', '3.0']
Score (numeric, 17 distinct): ['2.0', '2.2', '2.4', '2.6', '4.0', '4.2', '3.6', '3.8', '3.2', '4.4']
Inherent_Risk (numeric, 585 distinct): ['1.4', '1.578', '2.2', '1.418', '1.486', '1.568', '2.156', '1.8', '1.442', '1.414']
CONTROL_RISK (numeric, 12 distinct): ['0.4', '0.8', '1.2', '1.6', '2.4', '2.2', '2.0', '5.8', '3.4', '4.8']
Detection_Risk (numeric, 2 distinct): ['0.5']
Audit_Risk (numeric, 602 distinct): ['0.28', '0.3156', '1.32', '0.2836', '0.2828', '0.4312', '0.72', '0.2884', '0.2808', '0.3136']
audit (numeric, 2 distinct): ['1', '0']
trial (numeric, 2 distinct): ['0', '1']
SCORE_A (numeric, 4 distinct): ['2.0', '6.0', '4.0']
SCORE_B (numeric, 4 distinct): ['2.0', '6.0', '4.0']
Marks (numeric, 4 distinct): ['2.0', '4.0', '6.0']
MONEY_Marks (numeric, 4 distinct): ['2.0', '6.0', '4.0']
District (numeric, 4 distinct): ['2.0', '6.0', '4.0']
Loss (numeric, 4 distinct): ['0.0', '1.0', '2.0']
LOSS_SCORE (numeric, 4 distinct): ['2.0', '4.0', '6.0']
History_score (numeric, 4 distinct): ['2.0', '4.0', '6.0']
'''

CONTEXT = "Fraudulent Firm Classification"
TARGET = CuratedTarget(raw_name="Risk", new_name="Audit Risk", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "Fraudulent", "0": "Not Fraudulent"})
COLS_TO_DROP = []
FEATURES = []
