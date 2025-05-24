from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Satellite
====
Examples: 5100
====
URL: https://www.openml.org/search?type=data&id=40900
====
Description: **Author**: Markus Goldstein  
**Source**: [Dataverse](http://www.madm.eu/downloads https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF)  
**Please cite**:   

The satellite dataset comprises of features extracted from satellite observations. In particular, each image was taken under four different light wavelength, two in visible light (green and red) and two infrared images. The task of the original dataset is to classify the image into the soil category of the observed region. 

### Classes
We defined the soil classes &ldquo;red soil&rdquo;, &ldquo;gray soil&rdquo;, &ldquo;damp gray soil&rdquo; and &ldquo;very damp gray soil&rdquo; as the normal class. From the semantically different classes &ldquo;cotton crop&rdquo; and &ldquo;soil with vegetation stubble&rdquo; anomalies are sampled. 

After merging the original training and test set into a single dataset, the resulting dataset contains 5,025 normal instances as well as 75 randomly sampled anomalies (1.49%) with 36 dimensions 

### Relevant Papers

Goldstein, Markus, and Seiichi Uchida. A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data.&quot; PloS one 11.4 (2016): e0152173 

This dataset is not the original dataset. The target variable 'Target' is relabeled into 'Normal' and 'Anomaly'
====
Target Variable: Target (nominal, 2 distinct): ['Normal', 'Anomaly']
====
Features:

V1 (numeric, 49 distinct): ['67', '63', '88', '71', '68', '64', '84', '92', '70', '75']
V2 (numeric, 81 distinct): ['79', '75', '111', '103', '107', '99', '106', '91', '95', '83']
V3 (numeric, 69 distinct): ['114', '104', '100', '113', '96', '105', '118', '108', '82', '93']
V4 (numeric, 78 distinct): ['83', '87', '79', '92', '94', '96', '85', '81', '90', '76']
V5 (numeric, 48 distinct): ['67', '63', '71', '88', '68', '64', '84', '92', '70', '76']
V6 (numeric, 76 distinct): ['79', '75', '111', '103', '107', '99', '106', '91', '95', '83']
V7 (numeric, 68 distinct): ['104', '114', '100', '113', '96', '105', '118', '82', '93', '108']
V8 (numeric, 76 distinct): ['83', '87', '79', '92', '96', '94', '85', '81', '90', '76']
V9 (numeric, 47 distinct): ['67', '63', '71', '88', '68', '64', '84', '70', '75', '66']
V10 (numeric, 77 distinct): ['79', '75', '111', '103', '107', '99', '106', '91', '95', '83']
V11 (numeric, 69 distinct): ['104', '114', '100', '96', '113', '105', '82', '118', '108', '93']
V12 (numeric, 81 distinct): ['83', '87', '79', '92', '94', '96', '85', '81', '90', '76']
V13 (numeric, 49 distinct): ['67', '63', '71', '88', '68', '64', '84', '92', '70', '66']
V14 (numeric, 77 distinct): ['79', '75', '111', '103', '107', '99', '91', '95', '106', '87']
V15 (numeric, 67 distinct): ['104', '114', '100', '113', '96', '118', '105', '82', '108', '93']
V16 (numeric, 73 distinct): ['83', '87', '92', '79', '96', '94', '85', '81', '90', '78']
V17 (numeric, 49 distinct): ['67', '63', '71', '88', '68', '64', '84', '70', '75', '66']
V18 (numeric, 72 distinct): ['79', '75', '103', '111', '107', '99', '91', '95', '83', '106']
V19 (numeric, 67 distinct): ['104', '114', '113', '96', '100', '105', '118', '108', '82', '93']
V20 (numeric, 75 distinct): ['83', '87', '92', '79', '96', '94', '85', '81', '90', '78']
V21 (numeric, 47 distinct): ['67', '63', '71', '68', '88', '64', '84', '66', '70', '92']
V22 (numeric, 73 distinct): ['79', '75', '111', '103', '107', '99', '91', '83', '95', '106']
V23 (numeric, 69 distinct): ['104', '114', '96', '113', '100', '105', '118', '82', '93', '108']
V24 (numeric, 84 distinct): ['87', '83', '79', '92', '96', '94', '85', '81', '90', '76']
V25 (numeric, 49 distinct): ['67', '63', '71', '68', '64', '88', '84', '66', '70', '75']
V26 (numeric, 78 distinct): ['79', '75', '111', '103', '99', '107', '87', '95', '91', '83']
V27 (numeric, 69 distinct): ['104', '100', '114', '113', '96', '105', '82', '118', '108', '93']
V28 (numeric, 78 distinct): ['83', '87', '92', '79', '96', '94', '85', '81', '90', '76']
V29 (numeric, 48 distinct): ['67', '63', '71', '88', '68', '64', '84', '70', '66', '75']
V30 (numeric, 78 distinct): ['79', '75', '111', '103', '99', '107', '91', '95', '87', '106']
V31 (numeric, 68 distinct): ['104', '114', '100', '96', '113', '105', '82', '108', '118', '93']
V32 (numeric, 78 distinct): ['83', '87', '92', '79', '96', '94', '85', '81', '90', '78']
V33 (numeric, 49 distinct): ['67', '63', '71', '68', '88', '64', '84', '66', '70', '75']
V34 (numeric, 76 distinct): ['75', '79', '103', '111', '107', '91', '95', '99', '83', '106']
V35 (numeric, 70 distinct): ['104', '114', '96', '100', '113', '82', '105', '93', '108', '118']
V36 (numeric, 80 distinct): ['83', '87', '92', '79', '96', '94', '85', '81', '90', '78']
'''

CONTEXT = "Anonymized Dataset: Satellite"
TARGET = CuratedTarget(raw_name="Target", new_name="Anomaly Detected", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []