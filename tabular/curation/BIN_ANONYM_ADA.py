from tabular.datasets.manual_curation_obj import CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ada
====
Examples: 4147
====
URL: https://www.openml.org/search?type=data&id=41156
====
Description: The goal of this challenge is to expose the research community to real world datasets of interest to 4Paradigm. All datasets are formatted in a uniform way, though the type of data might differ. The data are provided as preprocessed matrices, so that participants can focus on classification, although participants are welcome to use additional feature extraction procedures (as long as they do not violate any rule of the challenge). All problems are binary classification problems and are assessed with the normalized Area Under the ROC Curve (AUC) metric (i.e. 2*AUC-1).
                   The identity of the datasets and the type of data is concealed, though its structure is revealed. The final score in  phase 2 will be the average of rankings  on all testing datasets, a ranking will be generated from such results, and winners will be determined according to such ranking.
                   The tasks are constrained by a time budget. The Codalab platform provides computational resources shared by all participants. Each code submission will be exceuted in a compute worker with the following characteristics: 2Cores / 8G Memory / 40G SSD with Ubuntu OS. To ensure the fairness of the evaluation, when a code submission is evaluated, its execution time is limited in time.
                   http://automl.chalearn.org/data
                   

https://www.openml.org/search?type=data&status=active&id=1043&sort=runs

Description
Author: Isabelle Guyon
Source: Agnostic Learning vs. Prior Knowledge Challenge
Please cite: None

Dataset from the Agnostic Learning vs. Prior Knowledge Challenge (http://www.agnostic.inf.ethz.ch), which consisted of 5 different datasets (SYLVA, GINA, NOVA, HIVA, ADA). The purpose of the challenge was to check if the performance of domain-specific feature engineering (prior knowledge) can be met by algorithms that were trained on data without any domain-specific knowledge (agnostic). For the latter, the data was anonymised and preprocessed in a way that makes them uninterpretable.

This dataset contains the agnostic (smashed) version of a data set from the US census bureau for the time span June 2005 - September 2006. Similar data set on OpenML is called adult. The raw data from the census bureau is also known as the Adult database in the UCI machine-learning repository.

Topic
The task of ADA is to discover high revenue people from census data. This is a two-class classification problem. The raw data from the census bureau is known as the Adult database in the UCI machine-learning repository. It contains continuous, binary and categorical variables. The “prior knowledge track” has access to the original features and their identity. The agnostic track has access to a preprocessed numeric representation eliminating categorical variables.

Source
Original owners This data was extracted from the census bureau database found at http://www.census.gov/ftp/pub/DES/www/welcome.html Donor: Ronny Kohavi and Barry Becker, Data Mining and Visualization Silicon Graphics. e-mail: ronnyk@sgi.com for questions

Dataset from: http://www.agnostic.inf.ethz.ch/datasets.php

Preprocessing
In this documentation the organisers of the challenge describe the steps they performed to come up with the agnostic data. The 14 original attributes (features) include age, workclass, education, marital status, occupation, native country, etc. It contains continuous, binary and categorical features. This dataset is from the "agnostic learning track", i.e. has access to a preprocessed numeric representation eliminating categorical variables, but the identity of the features is not revealed.

Additional Info
This dataset contains samples from both training and validation datasets. Modified by TunedIT (converted to ARFF format).

Data type: non-sparse Number of features: 48 Number of examples and check-sums: Pos_ex Neg_ex Tot_ex Check_sum Train 1029 3118 4147 6798109.00 Valid 103 312 415 681151.00


====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

V1 (numeric, 2 distinct): ['0', '1']
V2 (numeric, 2 distinct): ['0', '1']
V3 (numeric, 2 distinct): ['0', '1']
V4 (numeric, 330 distinct): ['117.0', '133.0', '124.0', '126.0', '127.0', '115.0', '131.0', '129.0', '109.0', '72.0']
V5 (numeric, 2 distinct): ['0', '1']
V6 (numeric, 2 distinct): ['0', '1']
V7 (numeric, 2 distinct): ['0', '1']
V8 (numeric, 2 distinct): ['1', '0']
V9 (numeric, 2 distinct): ['0', '1']
V10 (numeric, 48 distinct): ['0.0', '436.0', '453.0', '433.0', '430.0', '383.0', '341.0', '459.0', '424.0', '399.0']
V11 (numeric, 2 distinct): ['0', '1']
V12 (numeric, 2 distinct): ['0', '1']
V13 (numeric, 2 distinct): ['0', '1']
V14 (numeric, 1 distinct): ['0']
V15 (numeric, 16 distinct): ['562.0', '624.0', '812.0', '874.0', '687.0', '437.0', '749.0', '375.0', '250.0', '312.0']
V16 (numeric, 2 distinct): ['1', '0']
V17 (numeric, 2 distinct): ['0', '1']
V18 (numeric, 2 distinct): ['0', '1']
V19 (numeric, 2 distinct): ['1', '0']
V20 (numeric, 2 distinct): ['0', '1']
V21 (numeric, 1 distinct): ['0']
V22 (numeric, 2 distinct): ['0', '1']
V23 (numeric, 2 distinct): ['0', '1']
V24 (numeric, 2 distinct): ['0', '1']
V25 (numeric, 69 distinct): ['400.0', '300.0', '411.0', '389.0', '366.0', '311.0', '477.0', '333.0', '255.0', '466.0']
V26 (numeric, 2 distinct): ['0', '1']
V27 (numeric, 2 distinct): ['0', '1']
V28 (numeric, 2 distinct): ['0', '1']
V29 (numeric, 2 distinct): ['0', '1']
V30 (numeric, 2 distinct): ['0', '1']
V31 (numeric, 2 distinct): ['0', '1']
V32 (numeric, 51 distinct): ['0.0', '150.0', '77.0', '73.0', '999.0', '22.0', '31.0', '44.0', '50.0', '52.0']
V33 (numeric, 2 distinct): ['0', '1']
V34 (numeric, 2 distinct): ['0', '1']
V35 (numeric, 2 distinct): ['0', '1']
V36 (numeric, 2 distinct): ['0', '1']
V37 (numeric, 2 distinct): ['0', '1']
V38 (numeric, 2 distinct): ['0', '1']
V39 (numeric, 2 distinct): ['0', '1']
V40 (numeric, 75 distinct): ['404.0', '505.0', '454.0', '605.0', '202.0', '303.0', '353.0', '555.0', '252.0', '484.0']
V41 (numeric, 2 distinct): ['0', '1']
V42 (numeric, 2 distinct): ['0', '1']
V43 (numeric, 2 distinct): ['0', '1']
V44 (numeric, 2 distinct): ['0', '1']
V45 (numeric, 2 distinct): ['0', '1']
V46 (numeric, 2 distinct): ['0', '1']
V47 (numeric, 2 distinct): ['0', '1']
V48 (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Anonymized Dataset: Ada"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []