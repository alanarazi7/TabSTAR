from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: spoken-arabic-digit
====
Examples: 263256
====
URL: https://www.openml.org/search?type=data&id=1503
====
Description: **Author**: Data Collected by the Laboratory of Automatic and Signals, University of Badji-Mokhtar Annaba, Algeria.    
**Source**: UCI    
**Please cite**:   


* Title of Database: Spoken Arabic Digit

* Abstract: 

This dataset contains time series of mel-frequency cepstrum coefficients (MFCCs) corresponding to spoken Arabic digits. Includes data from 44 males and 44 females native Arabic speakers.

* Source:

Data Collected by the Laboratory of Automatic and Signals, 
University of Badji-Mokhtar 
Annaba, Algeria. 

Direction: Prof.Mouldi Bedda 
Participants: H.Dahmani, C.Snani, MC.Amara Korba, S.Atoui 
Adapted and preprocessed by : 
Nacereddine Hammami and Mouldi Bedda 
Faculty of Engineering, 
Al-Jouf University 
Sakaka, Al-Jouf 
Kingdom of Saudi Arabia 
e-mail: nacereddine.hammami '@' gmail.com 
mouldi_bedda '@' yahoo.fr 
Date: October, 2008


* Data Set Information:

Dataset from 8800 (10 digits x 10 repetitions x 88 speakers) time series of 13 Frequency Cepstral 
Coefficients (MFCCs) had taken from 44 males and 44 females Arabic native speakers 
between the ages 18 and 40 to represent ten spoken Arabic digit.


* Attribute Information:

Each line on the data base represents 13 MFCCs coefficients in the increasing order separated by spaces. This corresponds to one analysis frame. The 13 Mel Frequency Cepstral Coefficients (MFCCs) are computed with the following conditions; Sampling rate: 11025 Hz, 16 bits Window applied: hamming Filter pre-emphasized: 1-0.97Z^(-1)


* Relevant Papers:

[1] N. Hammami, M. Bedda ,"Improved Tree model for Arabic Speech Recognition", Proc. IEEE 
ICCSIT10 Conference, 2010. 
[2] N. Hammami, M. Sellami ,"Tree distribution classifier for automatic spoken Arabic digit 
recognition", Proc. IEEE ICITST09 Conference, 2009 , PP 1-4.
====
Target Variable: Class (nominal, 10 distinct): ['1', '6', '5', '10', '2', '4', '9', '8', '3', '7']
====
Features:

V1 (numeric, 119270 distinct): ['1.6458', '1.8851', '4.9308', '3.682', '2.766', '3.1172', '2.1179', '3.6446', '3.345', '2.9503']
V2 (numeric, 91022 distinct): ['-2.6961', '-2.6596', '-2.6698', '-2.484', '-2.7968', '-2.6274', '-2.6122', '-2.4898', '-2.5502', '-2.4279']
V3 (numeric, 138180 distinct): ['1.1143', '1.1037', '1.0817', '1.1911', '1.1651', '1.0044', '1.0758', '1.1254', '1.159', '1.0249']
V4 (numeric, 123326 distinct): ['-2.3952', '-1.2464', '-1.1154', '-2.3217', '-1.0126', '-1.6202', '-1.5275', '-1.0205', '-1.4864', '-1.929']
V5 (numeric, 135970 distinct): ['-1.3464', '-1.1357', '-1.0129', '-1.0089', '-1.125', '-1.145', '-1.228', '-1.4794', '-1.1878', '-1.1613']
V6 (numeric, 137186 distinct): ['-1.0468', '-1.1915', '-1.1014', '-1.2561', '-1.1644', '-1.0128', '-1.0853', '-1.0912', '-1.282', '-1.3758']
V7 (numeric, 151115 distinct): ['-1.0606', '-1.0071', '-1.0406', '-1.0321', '-1.0397', '-1.007', '-1.0368', '-1.2699', '-1.0506', '-1.1184']
V8 (numeric, 146786 distinct): ['-1.0323', '-1.0169', '-1.0963', '-1.0373', '-1.0155', '-1.0698', '-1.1572', '-1.1395', '-1.0544', '-1.0367']
V9 (numeric, 156467 distinct): ['-1.1637', '-1.0114', '-1.1035', '-1.0054', '-1.0564', '-1.0247', '-1.0756', '-1.1089', '-1.1334', '-1.0051']
V10 (numeric, 142862 distinct): ['-1.0057', '-1.0088', '-1.0144', '-1.0954', '-1.0704', '-1.0661', '-1.0461', '-1.0282', '-1.0195', '-1.073']
V11 (numeric, 150634 distinct): ['-1.0522', '-1.0545', '-1.064', '-1.0093', '-1.0728', '-1.0218', '-1.0341', '-1.0202', '-1.0019', '-1.0708']
V12 (numeric, 155756 distinct): ['-1.1241', '-1.0737', '-1.0094', '-1.0523', '-1.0114', '-1.0427', '-1.1088', '-1.0227', '-1.009', '-1.0366']
V13 (numeric, 155350 distinct): ['-1.0031', '-1.0199', '-1.1481', '-1.0056', '-1.0289', '-1.021', '-1.0057', '-0.455', '-1.0094', '-1.0553']
V14 (numeric, 2 distinct): ['2', '1']
'''

CONTEXT = "Spoken Arabic Digit Recognition relying on MFCCs"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []