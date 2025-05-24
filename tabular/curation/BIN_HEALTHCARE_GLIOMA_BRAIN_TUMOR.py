from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: glioma_grading_clinical_and_mutation_features
====
Examples: 839
====
URL: https://www.openml.org/search?type=data&id=46604
====
Description: For what purpose was the dataset created?

Gliomas are the most common primary tumors of the brain. They can be graded as LGG (Lower-Grade Glioma) or GBM (Glioblastoma Multiforme) depending on the histological/imaging criteria. Clinical and molecular/mutation factors are also very crucial for the grading process. Molecular tests are expensive to help accurately diagnose glioma patients.  

In this dataset, the most frequently mutated 20 genes and 3 clinical features are considered from TCGA-LGG and TCGA-GBM brain glioma projects.

The prediction task is to determine whether a patient is LGG or GBM with a given clinical and molecular/mutation features. The main objective is to find the optimal subset of mutation genes and clinical features for the glioma grading process to improve performance and reduce costs.

Who funded the creation of the dataset?

The Cancer Genome Atlas (TCGA) Project - NCI

What do the instances in this dataset represent?

In this dataset, the instances represent the records of patients who have brain glioma. The dataset was  constructed based on TCGA-LGG and TCGA-GBM brain glioma projects.

Each record is characterized by 20 molecular features (each of which can be mutated or not_mutated (wildtype) depending on the TCGA Case_ID) and 3 clinical features (concerning the demographics of the patient).

Are there recommended data splits?

No. We suggest 10-fold cross-validation for feature selection, classification etc.

Does the dataset contain data that might be considered sensitive in any way?

There is information about race, age, and gender of the patient.

Was there any data preprocessing performed?

Yes. 

The original and preprocessed files differ in the following ways:
- There are 23 instances in the original file where Gender, Age_at_diagnosis, or Race feature values are '--', or 'not reported'. These instances were filtered out in the preprocessed dataset.
- Despite being present in the original dataset, we do not include the columns Project, Case_ID, and Primary_Diagnosis columns in the preprocessed dataset.
- Age_at_diagnosis feature values were converted from string to continuous value by adding day information to the corresponding year information in the dataset as a floating-point number for the preprocessing stage.

All processed and unprocessed files also exist in this directory. 

Below is a list of the additional columns of the original dataset file (and their corresponding description):
- Project column represents corresponding TCGA-LGG or TCGA-GBM project names.
- Case_ID column refers to the related project Case_ID information.
- Primary_Diagnosis column provides information related to the type of primary diagnosis. 

Glioma grade class information (0 = "LGG"; 1 = "GBM")
====
Target Variable: Grade (numeric, 2 distinct): ['0', '1']
====
Features:

Gender (numeric, 2 distinct): ['0', '1']
Age_at_diagnosis (numeric, 766 distinct): ['52.67', '35.19', '58.04', '41.32', '72.53', '64.82', '55.13', '57.55', '48.95', '30.9']
Race (string, 4 distinct): ['white', 'black or african american', 'asian', 'american indian or alaska native']
IDH1 (numeric, 2 distinct): ['0', '1']
TP53 (numeric, 2 distinct): ['0', '1']
ATRX (numeric, 2 distinct): ['0', '1']
PTEN (numeric, 2 distinct): ['0', '1']
EGFR (numeric, 2 distinct): ['0', '1']
CIC (numeric, 2 distinct): ['0', '1']
MUC16 (numeric, 2 distinct): ['0', '1']
PIK3CA (numeric, 2 distinct): ['0', '1']
NF1 (numeric, 2 distinct): ['0', '1']
PIK3R1 (numeric, 2 distinct): ['0', '1']
FUBP1 (numeric, 2 distinct): ['0', '1']
RB1 (numeric, 2 distinct): ['0', '1']
NOTCH1 (numeric, 2 distinct): ['0', '1']
BCOR (numeric, 2 distinct): ['0', '1']
CSMD3 (numeric, 2 distinct): ['0', '1']
SMARCA4 (numeric, 2 distinct): ['0', '1']
GRIN2A (numeric, 2 distinct): ['0', '1']
IDH2 (numeric, 2 distinct): ['0', '1']
FAT4 (numeric, 2 distinct): ['0', '1']
PDGFRA (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Glioma Brain Tumor Grading"
TARGET = CuratedTarget(raw_name="Grade", task_type=SupervisedTask.BINARY,
                       label_mapping={"0": "LGG", "1": "GBM"})
COLS_TO_DROP = []
FEATURES = []