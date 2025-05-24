from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: QSAR_Bioconcentration_classification
====
Examples: 779
====
URL: https://www.openml.org/search?type=data&id=46585
====
Description: the QSAR Bioconcentration Classes Dataset is a well-known dataset used in cheminformatics and environmental chemistry. It is available from the UCI Machine Learning Repository and is often used for classification and regression tasks related to predicting the bioconcentration factor (BCF) of chemical compounds.

Dataset Overview
Objective: The primary goal of this dataset is to predict the bioconcentration factor (BCF) of chemical compounds, which is a measure of how likely a chemical is to accumulate in living organisms (e.g., fish) from the surrounding environment. This is important for assessing the environmental impact and toxicity of chemicals.

Features: The dataset contains molecular descriptors (features) that describe the chemical structure and properties of the compounds.

Target Variables:

logBCF: The logarithm of the bioconcentration factor. This is a continuous variable and is often used for regression tasks.

Class: A categorical variable that classifies compounds into different bioconcentration classes (e.g., low, medium, high). This is used for classification tasks.

Predict whether a chemical: (1) is mainly stored within lipid tissues, (2) has additional storage sites (e.g. proteins), or (3) is metabolized/eliminated. 

This dataset to OpenML is to predict Class, the column logBCF is not included in the dataset.
====
Target Variable: Class (numeric, 3 distinct): ['1', '3', '2']
====
Features:

CAS (string, 779 distinct): ['100-02-7', '626-67-5', '610-39-9', '611-06-3', '611-21-2', '61213-25-0', '612-22-6', '6130-75-2', '613-12-7', '61328-45-8']
SMILES (string, 779 distinct): ['O=[N+](c1ccc(cc1)O)[O-]', 'CN1CCCCC1', 'O=[N+](c1ccc(cc1[N+](=O)[O-])C)[O-]', 'O=[N+]([O-])c1ccc(cc1Cl)Cl', 'Cc1ccccc1NC', 'O=C2N(c1cccc(c1)C(F)(F)F)CC(CCl)C2Cl', 'CCc1c(cccc1)[N+](=O)[O-]', 'O(c1cc(c(cc1Cl)Cl)Cl)C', 'Cc1cc2c(cc3c(c2)cccc3)cc1', 'Clc1c(cc(c(c1)Cl)Cl)Oc1ccc(cc1)Cl']
Set (string, 2 distinct): ['Train', 'Test']
nHM (numeric, 11 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
piPC09 (numeric, 322 distinct): ['0.0', '5.901', '6.711', '5.614', '3.446', '3.726', '6.124', '6.423', '4.275', '6.019']
PCD (numeric, 224 distinct): ['0.0', '1.24', '1.23', '1.25', '1.26', '2.38', '1.49', '2.31', '1.27', '2.34']
X2Av (numeric, 63 distinct): ['0.21', '0.19', '0.2', '0.18', '0.17', '0.16', '0.15', '0.23', '0.22', '0.14']
MLOGP (numeric, 346 distinct): ['5.99', '2.73', '6.23', '2.19', '3.62', '3.31', '2.24', '1.7', '5.47', '2.89']
ON1V (numeric, 261 distinct): ['1.23', '1.21', '0.92', '1.25', '1.24', '1.2', '0.64', '0.63', '1.46', '1.06']
N-072 (numeric, 4 distinct): ['0', '1', '2', '3']
B02[C-N] (numeric, 2 distinct): ['0', '1']
F04[C-O] (numeric, 23 distinct): ['0', '4', '2', '1', '5', '3', '6', '8', '10', '7']
'''

CONTEXT = "QSAR Bioconcentration Classification"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.MULTICLASS,
                       new_name="Chemical Status",
                       label_mapping={
                           "1": "Stored in Lipid Tissues",
                           "2": "Metabolized/Eliminated",
                           "3": "Additional Storage Sites"
                       },)
COLS_TO_DROP = []
FEATURES = []