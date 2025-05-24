from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: arrhythmia
====
Examples: 452
====
URL: https://www.openml.org/search?type=data&id=5
====
Description: **Author**: H. Altay Guvenir, Burak Acar, Haldun Muderrisoglu  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/arrhythmia)   
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

**Cardiac Arrhythmia Database**  
The aim is to determine the type of arrhythmia from the ECG recordings. This database contains 279 attributes, 206 of which are linear valued and the rest are nominal. 

Concerning the study of H. Altay Guvenir: "The aim is to distinguish between the presence and absence of cardiac arrhythmia and to classify it in one of the 16 groups. Class 01 refers to 'normal' ECG classes, 02 to 15 refers to different classes of arrhythmia and class 16 refers to the rest of unclassified ones. For the time being, there exists a computer program that makes such a classification. However, there are differences between the cardiologist's and the program's classification. Taking the cardiologist's as a gold standard we aim to minimize this difference by means of machine learning tools.
 
The names and id numbers of the patients were recently removed from the database.
 
### Attribute Information  
 
       1 Age: Age in years , linear
       2 Sex: Sex (0 = male; 1 = female) , nominal
       3 Height: Height in centimeters , linear
       4 Weight: Weight in kilograms , linear
       5 QRS duration: Average of QRS duration in msec., linear
       6 P-R interval: Average duration between onset of P and Q waves
         in msec., linear
       7 Q-T interval: Average duration between onset of Q and offset
         of T waves in msec., linear
       8 T interval: Average duration of T wave in msec., linear
       9 P interval: Average duration of P wave in msec., linear
      Vector angles in degrees on front plane of:, linear
      10 QRS
      11 T
      12 P
      13 QRST
      14 J
      15 Heart rate: Number of heart beats per minute ,linear
      Of channel DI:
       Average width, in msec., of: linear
       16 Q wave
       17 R wave
       18 S wave
       19 R' wave, small peak just after R
       20 S' wave
       21 Number of intrinsic deflections, linear
       22 Existence of ragged R wave, nominal
       23 Existence of diphasic derivation of R wave, nominal
       24 Existence of ragged P wave, nominal
       25 Existence of diphasic derivation of P wave, nominal
       26 Existence of ragged T wave, nominal
       27 Existence of diphasic derivation of T wave, nominal
      Of channel DII: 
       28 .. 39 (similar to 16 .. 27 of channel DI)
      Of channels DIII:
       40 .. 51
      Of channel AVR:
       52 .. 63
      Of channel AVL:
       64 .. 75
      Of channel AVF:
       76 .. 87
      Of channel V1:
       88 .. 99
      Of channel V2:
       100 .. 111
      Of channel V3:
       112 .. 123
      Of channel V4:
       124 .. 135
      Of channel V5:
       136 .. 147
      Of channel V6:
       148 .. 159
      Of channel DI:
       Amplitude , * 0.1 milivolt, of
       160 JJ wave, linear
       161 Q wave, linear
       162 R wave, linear
       163 S wave, linear
       164 R' wave, linear
       165 S' wave, linear
       166 P wave, linear
       167 T wave, linear
       168 QRSA , Sum of areas of all segments divided by 10,
           ( Area= width * height / 2 ), linear
       169 QRSTA = QRSA + 0.5 * width of T wave * 0.1 * height of T
           wave. (If T is diphasic then the bigger segment is
           considered), linear
      Of channel DII:
       170 .. 179
      Of channel DIII:
       180 .. 189
      Of channel AVR:
       190 .. 199
      Of channel AVL:
       200 .. 209
      Of channel AVF:
       210 .. 219
      Of channel V1:
       220 .. 229
      Of channel V2:
       230 .. 239
      Of channel V3:
       240 .. 249
      Of channel V4:
       250 .. 259
      Of channel V5:
       260 .. 269
      Of channel V6:
       270 .. 279
        
Class code - class - number of instances:
> 
        01             Normal                245
        02             Ischemic changes (Coronary Artery Disease)   44
        03             Old Anterior Myocardial Infarction           15
        04             Old Inferior Myocardial Infarction           15
        05             Sinus tachycardy        13
        06             Sinus bradycardy        25
        07             Ventricular Premature Contraction (PVC)       3
        08             Supraventricular Premature Contraction       2
        09             Left bundle branch block         9 
        10             Right bundle branch block       50
        11             1. degree AtrioVentricular block       0 
        12             2. degree AV block                0
        13             3. degree AV block                0
        14             Left ventricule hypertrophy                4
        15             Atrial Fibrillation or Flutter               5
        16             Others                 22
====
Target Variable: class (nominal, 13 distinct): ['1', '10', '2', '6', '16', '3', '4', '5', '9', '15']
====
Features:

age (numeric, 77 distinct): ['46', '37', '36', '47', '35', '44', '45', '50', '57', '40']
sex (nominal, 2 distinct): ['1', '0']
height (numeric, 53 distinct): ['160.0', '165.0', '170.0', '155.0', '175.0', '156.0', '163.0', '162.0', '168.0', '172.0']
weight (numeric, 76 distinct): ['80', '70', '65', '60', '75', '55', '74', '68', '85', '72']
QRSduration (numeric, 67 distinct): ['82', '85', '90', '80', '78', '84', '87', '91', '81', '83']
PRinterval (numeric, 106 distinct): ['0.0', '157.0', '145.0', '164.0', '155.0', '160.0', '163.0', '147.0', '154.0', '158.0']
Q-Tinterval (numeric, 132 distinct): ['383.0', '357.0', '364.0', '362.0', '359.0', '386.0', '377.0', '382.0', '350.0', '376.0']
Tinterval (numeric, 129 distinct): ['152.0', '172.0', '163.0', '156.0', '154.0', '169.0', '137.0', '149.0', '147.0', '160.0']
Pinterval (numeric, 90 distinct): ['100', '96', '82', '83', '80', '104', '81', '0', '93', '97']
QRS (numeric, 160 distinct): ['62.0', '66.0', '56.0', '-1.0', '33.0', '52.0', '64.0', '25.0', '60.0', '50.0']
T (numeric, 171 distinct): ['52.0', '36.0', '42.0', '33.0', '48.0', '10.0', '41.0', '13.0', '14.0', '19.0']
P (numeric, 102 distinct): ['60.0', '61.0', '56.0', '58.0', '68.0', '50.0', '63.0', '52.0', '74.0', '55.0']
QRST (numeric, 135 distinct): ['62.0', '59.0', '49.0', '55.0', '33.0', '38.0', '40.0', '26.0', '63.0', '12.0']
J (numeric, 70 distinct): ['84.0', '169.0', '-93.0', '103.0', '-157.0', '-164.0', '23.0', '-119.0', '144.0', '123.0']
heartrate (numeric, 64 distinct): ['63.0', '72.0', '70.0', '73.0', '81.0', '68.0', '78.0', '75.0', '80.0', '71.0']
chDI_Qwave (numeric, 11 distinct): ['0', '20', '16', '24', '12', '28', '44', '32', '36', '88']
chDI_Rwave (numeric, 28 distinct): ['44', '40', '48', '36', '52', '56', '60', '64', '68', '72']
chDI_Swave (numeric, 20 distinct): ['0', '36', '40', '32', '48', '44', '28', '24', '20', '16']
chDI_RPwave (numeric, 4 distinct): ['0', '12', '24', '16']
chDI_SPwave (numeric, 1 distinct): ['0']
chDI_intrinsicReflecttions (numeric, 18 distinct): ['24', '28', '20', '36', '32', '40', '44', '48', '16', '56']
chDI_RRwaveExists (nominal, 2 distinct): ['0', '1']
chDI_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chDI_RPwaveExists (nominal, 2 distinct): ['0', '1']
chDI_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chDI_RTwaveExists (nominal, 2 distinct): ['0', '1']
chDI_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chDII_Qwave (numeric, 11 distinct): ['0', '20', '24', '16', '28', '36', '12', '40', '32', '76']
chDII_Rwave (numeric, 26 distinct): ['48', '44', '40', '52', '56', '60', '72', '68', '64', '36']
chDII_Swave (numeric, 20 distinct): ['0', '36', '40', '32', '24', '28', '48', '44', '20', '52']
chDII_RPwave (numeric, 7 distinct): ['0', '20', '16', '36', '28', '12', '8']
chDII_SPwave (numeric, 3 distinct): ['0', '12', '56']
chDII_intrinsicReflecttions (numeric, 17 distinct): ['28', '24', '32', '40', '36', '44', '20', '48', '52', '16']
chDII_RRwaveExists (nominal, 2 distinct): ['0', '1']
chDII_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chDII_RPwaveExists (nominal, 2 distinct): ['0', '1']
chDII_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chDII_RTwaveExists (nominal, 2 distinct): ['0', '1']
chDII_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chDIII_Qwave (numeric, 22 distinct): ['0', '20', '24', '32', '28', '36', '16', '48', '44', '40']
chDIII_Rwave (numeric, 25 distinct): ['0', '40', '44', '48', '24', '32', '28', '52', '20', '36']
chDIII_Swave (numeric, 22 distinct): ['0', '44', '40', '36', '28', '52', '56', '20', '32', '24']
chDIII_RPwave (numeric, 16 distinct): ['0', '16', '28', '12', '52', '48', '20', '36', '24', '40']
chDIII_SPwave (numeric, 4 distinct): ['0', '24', '44', '28']
chDIII_intrinsicReflecttions (numeric, 22 distinct): ['0', '44', '24', '40', '16', '28', '20', '36', '32', '52']
chDIII_RRwaveExists (nominal, 2 distinct): ['0', '1']
chDIII_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chDIII_RPwaveExists (nominal, 2 distinct): ['0', '1']
chDIII_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chDIII_RTwaveExists (nominal, 2 distinct): ['0', '1']
chDIII_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chAVR_Qwave (numeric, 24 distinct): ['48', '0', '44', '52', '40', '56', '64', '72', '76', '68']
chAVR_Rwave (numeric, 17 distinct): ['0', '20', '36', '28', '24', '16', '32', '40', '44', '48']
chAVR_Swave (numeric, 15 distinct): ['0', '44', '48', '36', '40', '68', '52', '56', '28', '32']
chAVR_RPwave (numeric, 14 distinct): ['0', '32', '36', '20', '48', '40', '28', '16', '44', '64']
chAVR_SPwave (numeric, 2 distinct): ['0', '32']
chAVR_intrinsicReflecttions (numeric, 21 distinct): ['0', '48', '52', '56', '60', '44', '64', '8', '12', '68']
chAVR_RRwaveExists (nominal, 2 distinct): ['0', '1']
chAVR_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chAVR_RPwaveExists (nominal, 2 distinct): ['0', '1']
chAVR_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chAVR_RTwaveExists (nominal, 2 distinct): ['0', '1']
chAVR_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chAVL_Qwave (numeric, 20 distinct): ['0', '20', '24', '28', '16', '32', '52', '36', '44', '12']
chAVL_Rwave (numeric, 29 distinct): ['36', '32', '40', '0', '44', '56', '28', '52', '60', '48']
chAVL_Swave (numeric, 22 distinct): ['0', '40', '36', '32', '28', '48', '44', '20', '60', '56']
chAVL_RPwave (numeric, 7 distinct): ['0', '32', '24', '36', '40', '28', '44']
chAVL_SPwave (numeric, 1 distinct): ['0']
chAVL_intrinsicReflecttions (numeric, 22 distinct): ['24', '20', '16', '28', '36', '32', '0', '40', '12', '44']
chAVL_RRwaveExists (nominal, 1 distinct): ['0', '1']
chAVL_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chAVL_RPwaveExists (nominal, 2 distinct): ['0', '1']
chAVL_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chAVL_RTwaveExists (nominal, 2 distinct): ['0', '1']
chAVL_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chAVF_Qwave (numeric, 19 distinct): ['0', '20', '24', '16', '32', '44', '28', '36', '48', '12']
chAVF_Rwave (numeric, 25 distinct): ['48', '44', '40', '56', '36', '64', '52', '32', '76', '60']
chAVF_Swave (numeric, 23 distinct): ['0', '32', '40', '36', '44', '48', '52', '20', '16', '28']
chAVF_RPwave (numeric, 10 distinct): ['0', '20', '36', '16', '28', '24', '32', '44', '12', '8']
chAVF_SPwave (numeric, 4 distinct): ['0', '24', '44', '32']
chAVF_intrinsicReflecttions (numeric, 21 distinct): ['28', '32', '24', '20', '40', '36', '44', '16', '12', '48']
chAVF_RRwaveExists (nominal, 2 distinct): ['0', '1']
chAVF_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chAVF_RPwaveExists (nominal, 1 distinct): ['0', '1']
chAVF_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chAVF_RTwaveExists (nominal, 2 distinct): ['0', '1']
chAVF_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV1_Qwave (numeric, 21 distinct): ['0', '60', '56', '64', '76', '48', '72', '68', '52', '80']
chV1_Rwave (numeric, 17 distinct): ['24', '28', '0', '32', '20', '36', '40', '16', '48', '44']
chV1_Swave (numeric, 23 distinct): ['0', '60', '56', '52', '48', '44', '64', '40', '68', '36']
chV1_RPwave (numeric, 17 distinct): ['0', '36', '32', '44', '28', '40', '20', '12', '24', '52']
chV1_SPwave (numeric, 3 distinct): ['0', '12', '28']
chV1_intrinsicReflecttions (numeric, 25 distinct): ['16', '12', '0', '20', '24', '8', '28', '72', '4', '76']
chV1_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV1_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV1_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV1_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV1_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV1_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV2_Qwave (numeric, 21 distinct): ['0', '60', '72', '84', '20', '64', '88', '48', '68', '80']
chV2_Rwave (numeric, 18 distinct): ['36', '32', '40', '44', '28', '24', '0', '48', '20', '52']
chV2_Swave (numeric, 22 distinct): ['44', '52', '48', '56', '40', '0', '60', '36', '64', '32']
chV2_RPwave (numeric, 14 distinct): ['0', '20', '28', '24', '44', '36', '16', '48', '60', '8']
chV2_SPwave (numeric, 3 distinct): ['0', '32', '16']
chV2_intrinsicReflecttions (numeric, 23 distinct): ['20', '24', '16', '28', '0', '32', '12', '36', '40', '8']
chV2_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV2_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV2_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV2_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV2_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV2_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV3_Qwave (numeric, 14 distinct): ['0', '80', '24', '20', '16', '84', '88', '64', '76', '92']
chV3_Rwave (numeric, 19 distinct): ['44', '40', '48', '52', '36', '32', '56', '0', '60', '28']
chV3_Swave (numeric, 26 distinct): ['44', '40', '48', '36', '32', '52', '56', '28', '0', '60']
chV3_RPwave (numeric, 6 distinct): ['0', '20', '56', '24', '76', '48']
chV3_SPwave (numeric, 3 distinct): ['0', '24', '36']
chV3_intrinsicReflecttions (numeric, 16 distinct): ['24', '28', '32', '36', '20', '40', '16', '0', '44', '12']
chV3_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV3_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV3_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV3_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV3_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV3_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV4_Qwave (numeric, 13 distinct): ['0', '20', '24', '16', '12', '28', '80', '88', '84', '56']
chV4_Rwave (numeric, 19 distinct): ['44', '48', '52', '40', '36', '56', '60', '32', '64', '0']
chV4_Swave (numeric, 28 distinct): ['40', '44', '36', '48', '52', '56', '24', '60', '20', '32']
chV4_RPwave (numeric, 6 distinct): ['0', '16', '20', '8', '24', '12']
chV4_SPwave (numeric, 3 distinct): ['0', '40', '16']
chV4_intrinsicReflecttions (numeric, 14 distinct): ['32', '28', '36', '24', '40', '44', '20', '48', '0', '16']
chV4_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV4_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV4_RPwaveExists (nominal, 1 distinct): ['0', '1']
chV4_DD_RPwaveExists (nominal, 1 distinct): ['0', '1']
chV4_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV4_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV5_Qwave (numeric, 10 distinct): ['0', '20', '16', '24', '28', '12', '80', '36', '32', '88']
chV5_Rwave (numeric, 23 distinct): ['48', '44', '40', '52', '36', '56', '32', '60', '64', '68']
chV5_Swave (numeric, 23 distinct): ['40', '36', '44', '48', '56', '0', '32', '52', '28', '20']
chV5_RPwave (numeric, 4 distinct): ['0', '16', '60', '20']
chV5_SPwave (numeric, 1 distinct): ['0']
chV5_intrinsicReflecttions (numeric, 18 distinct): ['28', '32', '24', '40', '36', '44', '20', '48', '52', '16']
chV5_RRwaveExists (nominal, 1 distinct): ['0', '1']
chV5_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV5_RPwaveExists (nominal, 1 distinct): ['0', '1']
chV5_DD_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV5_RTwaveExists (nominal, 1 distinct): ['0', '1']
chV5_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chV6_Qwave (numeric, 11 distinct): ['0', '20', '16', '24', '12', '28', '32', '88', '68', '36']
chV6_Rwave (numeric, 25 distinct): ['48', '44', '52', '40', '36', '56', '60', '68', '64', '76']
chV6_Swave (numeric, 20 distinct): ['0', '36', '40', '44', '48', '52', '32', '56', '20', '28']
chV6_RPwave (numeric, 3 distinct): ['0', '20', '28']
chV6_SPwave (numeric, 1 distinct): ['0']
chV6_intrinsicReflecttions (numeric, 17 distinct): ['28', '24', '32', '40', '36', '44', '20', '48', '52', '60']
chV6_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV6_DD_RRwaveExists (nominal, 2 distinct): ['0', '1']
chV6_RPwaveExists (nominal, 2 distinct): ['0', '1']
chV6_DD_RPwaveExists (nominal, 1 distinct): ['0', '1']
chV6_RTwaveExists (nominal, 1 distinct): ['0', '1']
chV6_DD_RTwaveExists (nominal, 2 distinct): ['0', '1']
chDI_JJwaveAmp (numeric, 32 distinct): ['0.0', '-0.1', '0.1', '-0.2', '-0.5', '-0.3', '-0.4', '0.2', '0.4', '-0.6']
chDI_QwaveAmp (numeric, 17 distinct): ['0.0', '-0.4', '-0.6', '-0.5', '-0.7', '-0.8', '-1.0', '-0.9', '-2.1', '-1.1']
chDI_RwaveAmp (numeric, 111 distinct): ['4.0', '6.1', '4.4', '6.3', '7.8', '5.5', '3.6', '3.2', '3.8', '5.1']
chDI_SwaveAmp (numeric, 52 distinct): ['0.0', '-0.6', '-1.3', '-1.1', '-0.8', '-1.7', '-0.9', '-1.0', '-1.6', '-1.4']
chDI_RPwaveAmp (numeric, 3 distinct): ['0.0', '0.4', '1.9']
chDI_SPwaveAmp (numeric, 1 distinct): ['0']
chDI_PwaveAmp (numeric, 24 distinct): ['0.8', '0.6', '0.7', '0.5', '0.4', '0.9', '1.0', '0.3', '0.2', '1.1']
chDI_TwaveAmp (numeric, 56 distinct): ['1.3', '1.6', '1.2', '1.9', '0.8', '0.9', '1.0', '0.7', '1.5', '1.7']
chDI_QRSA (numeric, 268 distinct): ['13.4', '11.7', '7.5', '6.4', '8.0', '21.2', '23.1', '10.6', '6.0', '4.0']
chDI_QRSTA (numeric, 296 distinct): ['12.2', '17.3', '26.2', '35.5', '27.6', '23.3', '9.1', '27.9', '3.9', '35.4']
chDII_JJwaveAmp (numeric, 35 distinct): ['0.0', '0.1', '-0.1', '-0.2', '-0.4', '-0.3', '0.2', '-0.5', '0.4', '0.5']
chDII_QwaveAmp (numeric, 25 distinct): ['0.0', '-0.5', '-0.4', '-1.0', '-0.8', '-0.9', '-0.6', '-0.7', '-1.1', '-1.3']
chDII_RwaveAmp (numeric, 137 distinct): ['4.5', '6.4', '4.7', '7.0', '8.5', '9.8', '5.1', '7.1', '4.3', '4.0']
chDII_SwaveAmp (numeric, 58 distinct): ['0.0', '-1.3', '-1.7', '-2.6', '-1.0', '-1.4', '-1.6', '-1.5', '-0.7', '-0.9']
chDII_RPwaveAmp (numeric, 8 distinct): ['0.0', '0.4', '0.6', '0.7', '1.0', '3.2', '0.9', '0.8']
chDII_SPwaveAmp (numeric, 3 distinct): ['0.0', '-1.2', '-1.5']
chDII_PwaveAmp (numeric, 37 distinct): ['1.0', '1.1', '0.8', '0.9', '1.2', '0.6', '1.3', '1.4', '0.7', '1.5']
chDII_TwaveAmp (numeric, 68 distinct): ['0.6', '1.4', '1.8', '0.5', '0.8', '1.5', '1.6', '1.7', '1.2', '1.1']
chDII_QRSA (numeric, 290 distinct): ['15.9', '13.6', '6.7', '25.5', '22.4', '5.8', '17.6', '16.9', '12.4', '14.4']
chDII_QRSTA (numeric, 326 distinct): ['30.5', '21.2', '9.5', '30.7', '22.6', '24.0', '32.2', '12.6', '28.6', '43.9']
chDIII_JJwaveAmp (numeric, 34 distinct): ['0.0', '0.1', '-0.1', '-0.2', '0.2', '0.6', '-0.5', '0.4', '-0.4', '0.3']
chDIII_QwaveAmp (numeric, 62 distinct): ['0.0', '-0.5', '-0.6', '-0.4', '-0.9', '-1.8', '-0.7', '-1.5', '-1.0', '-0.8']
chDIII_RwaveAmp (numeric, 105 distinct): ['0.0', '0.5', '1.3', '0.9', '0.7', '0.4', '1.6', '3.3', '1.0', '1.4']
chDIII_SwaveAmp (numeric, 87 distinct): ['0.0', '-0.5', '-0.6', '-2.4', '-0.4', '-1.2', '-0.8', '-3.1', '-1.8', '-5.2']
chDIII_RPwaveAmp (numeric, 24 distinct): ['0.0', '0.4', '0.5', '1.3', '0.7', '2.3', '1.0', '3.1', '2.6', '2.4']
chDIII_SPwaveAmp (numeric, 6 distinct): ['0.0', '-1.3', '-0.7', '-1.1', '-1.7', '-1.2']
chDIII_PwaveAmp (numeric, 34 distinct): ['0.4', '0.7', '0.5', '0.6', '0.2', '0.3', '0.8', '0.9', '0.1', '1.0']
chDIII_TwaveAmp (numeric, 56 distinct): ['0.2', '0.6', '0.7', '-0.6', '0.4', '-0.5', '0.8', '-0.7', '0.5', '0.3']
chDIII_QRSA (numeric, 304 distinct): ['3.0', '8.7', '5.2', '-9.8', '-0.7', '9.6', '-13.0', '-9.3', '-3.3', '2.2']
chDIII_QRSTA (numeric, 318 distinct): ['7.4', '-6.5', '8.4', '-3.1', '1.8', '9.3', '33.9', '6.6', '-11.0', '10.4']
chAVR_JJwaveAmp (numeric, 31 distinct): ['0.1', '-0.1', '0.0', '0.2', '0.4', '-0.2', '0.3', '0.5', '-0.4', '0.7']
chAVR_QwaveAmp (numeric, 96 distinct): ['0.0', '-6.1', '-6.3', '-4.8', '-6.4', '-6.6', '-5.6', '-5.2', '-5.0', '-7.1']
chAVR_RwaveAmp (numeric, 40 distinct): ['0.0', '0.5', '0.4', '1.3', '0.7', '1.0', '0.6', '0.8', '1.1', '0.9']
chAVR_SwaveAmp (numeric, 53 distinct): ['0.0', '-5.3', '-7.4', '-5.2', '-10.6', '-6.5', '-5.9', '-3.5', '-4.7', '-9.2']
chAVR_RPwaveAmp (numeric, 21 distinct): ['0.0', '0.8', '0.6', '1.8', '1.0', '0.5', '2.5', '1.2', '2.4', '2.0']
chAVR_SPwaveAmp (numeric, 2 distinct): ['0.0', '-0.4']
chAVR_PwaveAmp (numeric, 25 distinct): ['-0.9', '-0.8', '-0.6', '-0.7', '-1.1', '-1.0', '-0.5', '-1.2', '-1.3', '-0.4']
chAVR_TwaveAmp (numeric, 60 distinct): ['-1.0', '-1.5', '-1.1', '-1.2', '-2.2', '-0.9', '-1.9', '-1.3', '-0.7', '-2.1']
chAVR_QRSA (numeric, 253 distinct): ['-11.5', '-14.4', '-12.4', '-13.6', '-9.8', '-13.2', '-16.3', '-24.1', '-11.7', '-8.3']
chAVR_QRSTA (numeric, 281 distinct): ['-22.2', '-14.7', '-29.6', '-14.3', '-16.4', '-19.8', '-15.1', '-18.3', '-23.1', '-18.1']
chAVL_JJwaveAmp (numeric, 31 distinct): ['0.0', '0.1', '-0.1', '-0.2', '-0.4', '0.2', '-0.3', '-0.5', '-0.6', '-0.7']
chAVL_QwaveAmp (numeric, 37 distinct): ['0.0', '-0.6', '-0.5', '-0.4', '-0.7', '-1.0', '-1.1', '-0.8', '-0.9', '-1.2']
chAVL_RwaveAmp (numeric, 101 distinct): ['0.0', '0.8', '0.4', '1.9', '2.0', '1.5', '1.0', '5.7', '1.1', '2.1']
chAVL_SwaveAmp (numeric, 62 distinct): ['0.0', '-0.9', '-1.1', '-1.4', '-1.9', '-1.0', '-2.5', '-1.3', '-2.0', '-1.5']
chAVL_RPwaveAmp (numeric, 9 distinct): ['0.0', '0.8', '0.9', '1.7', '2.3', '0.6', '1.0', '1.5', '1.4']
chAVL_SPwaveAmp (numeric, 1 distinct): ['0']
chAVL_PwaveAmp (numeric, 23 distinct): ['-0.1', '-0.2', '-0.3', '0.3', '0.4', '0.5', '0.6', '0.7', '0.2', '-0.4']
chAVL_TwaveAmp (numeric, 46 distinct): ['0.4', '0.5', '0.7', '0.3', '1.1', '0.8', '-0.3', '0.6', '1.2', '-0.2']
chAVL_QRSA (numeric, 276 distinct): ['-2.0', '3.7', '2.7', '14.1', '11.4', '0.3', '1.8', '0.7', '3.3', '10.5']
chAVL_QRSTA (numeric, 289 distinct): ['7.7', '-1.4', '-3.8', '8.5', '14.9', '12.3', '3.9', '3.0', '-6.7', '14.3']
chAVF_JJwaveAmp (numeric, 28 distinct): ['0.0', '-0.1', '0.1', '-0.2', '0.2', '0.4', '-0.4', '-0.5', '-0.3', '0.6']
chAVF_QwaveAmp (numeric, 34 distinct): ['0.0', '-0.4', '-0.6', '-0.7', '-0.9', '-1.2', '-0.5', '-1.1', '-1.0', '-1.4']
chAVF_RwaveAmp (numeric, 118 distinct): ['0.0', '4.4', '5.4', '2.7', '3.2', '0.4', '0.7', '2.5', '7.3', '4.8']
chAVF_SwaveAmp (numeric, 64 distinct): ['0.0', '-1.2', '-1.0', '-0.9', '-0.6', '-1.9', '-1.5', '-1.3', '-3.3', '-0.8']
chAVF_RPwaveAmp (numeric, 11 distinct): ['0.0', '0.5', '0.4', '0.7', '9.1', '0.8', '1.8', '1.6', '1.0', '3.9']
chAVF_SPwaveAmp (numeric, 4 distinct): ['0.0', '-0.9', '-4.5', '-3.3']
chAVF_PwaveAmp (numeric, 33 distinct): ['0.8', '0.7', '0.6', '1.0', '0.4', '0.9', '0.5', '0.3', '1.1', '1.3']
chAVF_TwaveAmp (numeric, 54 distinct): ['0.4', '0.5', '0.6', '0.3', '0.7', '0.8', '1.2', '1.3', '0.9', '1.4']
chAVF_QRSA (numeric, 279 distinct): ['7.2', '15.6', '2.1', '4.8', '10.2', '1.5', '-1.0', '3.4', '5.4', '17.1']
chAVF_QRSTA (numeric, 317 distinct): ['15.0', '22.8', '11.7', '14.1', '0.7', '-4.0', '4.3', '10.6', '24.5', '3.0']
chV1_JJwaveAmp (numeric, 45 distinct): ['0.1', '0.4', '0.5', '0.6', '0.9', '0.7', '0.2', '0.8', '1.0', '0.0']
chV1_QwaveAmp (numeric, 62 distinct): ['0.0', '-4.5', '-7.3', '-3.9', '-8.3', '-3.8', '-10.8', '-8.5', '-3.4', '-6.3']
chV1_RwaveAmp (numeric, 58 distinct): ['0.0', '0.7', '0.9', '1.1', '1.3', '0.5', '0.6', '1.2', '0.4', '1.0']
chV1_SwaveAmp (numeric, 129 distinct): ['0.0', '-6.5', '-5.3', '-9.0', '-5.4', '-7.7', '-8.5', '-8.3', '-5.1', '-6.1']
chV1_RPwaveAmp (numeric, 27 distinct): ['0.0', '1.8', '1.1', '0.5', '1.9', '1.3', '2.4', '1.0', '1.5', '2.7']
chV1_SPwaveAmp (numeric, 4 distinct): ['0.0', '-0.4', '-0.7', '-2.9']
chV1_PwaveAmp (numeric, 33 distinct): ['0.0', '-0.6', '-0.8', '0.1', '-0.4', '-0.7', '0.2', '-0.5', '0.3', '-0.3']
chV1_TwaveAmp (numeric, 72 distinct): ['-0.5', '0.7', '-0.3', '-0.4', '0.5', '-0.1', '-0.9', '-1.0', '1.0', '0.6']
chV1_QRSA (numeric, 287 distinct): ['-21.8', '-28.7', '-17.1', '-36.4', '-14.3', '-25.0', '-13.7', '-25.3', '-16.7', '-15.3']
chV1_QRSTA (numeric, 309 distinct): ['-13.2', '-18.2', '-18.3', '-25.0', '-12.7', '-4.4', '-17.4', '-15.8', '-11.4', '-11.9']
chV2_JJwaveAmp (numeric, 57 distinct): ['0.1', '0.6', '0.9', '1.1', '0.5', '0.7', '0.0', '0.8', '1.0', '0.4']
chV2_QwaveAmp (numeric, 38 distinct): ['0.0', '-0.5', '-6.7', '-4.7', '-0.4', '-4.9', '-4.5', '-26.6', '-8.7', '-12.2']
chV2_RwaveAmp (numeric, 106 distinct): ['0.0', '2.6', '2.3', '2.0', '3.5', '1.2', '1.8', '3.2', '3.0', '3.4']
chV2_SwaveAmp (numeric, 171 distinct): ['0.0', '-12.5', '-6.0', '-7.9', '-7.1', '-6.7', '-4.6', '-7.6', '-8.3', '-8.0']
chV2_RPwaveAmp (numeric, 24 distinct): ['0.0', '2.1', '3.0', '1.0', '1.3', '3.4', '0.9', '1.6', '2.7', '14.9']
chV2_SPwaveAmp (numeric, 4 distinct): ['0.0', '-2.1', '-1.1', '-4.0']
chV2_PwaveAmp (numeric, 28 distinct): ['0.1', '0.0', '0.3', '0.2', '-0.4', '-0.2', '0.4', '-0.3', '0.5', '-0.5']
chV2_TwaveAmp (numeric, 107 distinct): ['2.1', '1.4', '2.6', '3.4', '2.2', '1.6', '3.0', '1.1', '1.7', '4.0']
chV2_QRSA (numeric, 315 distinct): ['-13.8', '-25.5', '-16.8', '-6.7', '-1.4', '-14.9', '-13.6', '-15.6', '-27.9', '-16.4']
chV2_QRSTA (numeric, 350 distinct): ['17.9', '16.4', '35.1', '29.8', '7.6', '7.5', '2.5', '-5.2', '4.7', '1.3']
chV3_JJwaveAmp (numeric, 64 distinct): ['0.1', '0.0', '0.6', '0.7', '0.5', '0.2', '0.9', '0.8', '1.1', '-0.1']
chV3_QwaveAmp (numeric, 26 distinct): ['0.0', '-1.0', '-0.9', '-0.5', '-15.5', '-10.3', '-17.1', '-17.9', '-10.5', '-18.0']
chV3_RwaveAmp (numeric, 164 distinct): ['0.0', '2.4', '7.0', '3.6', '8.2', '6.4', '10.9', '5.6', '4.2', '5.4']
chV3_SwaveAmp (numeric, 189 distinct): ['0.0', '-10.0', '-5.8', '-5.4', '-6.6', '-6.8', '-9.9', '-10.3', '-8.4', '-9.1']
chV3_RPwaveAmp (numeric, 7 distinct): ['0.0', '4.0', '1.1', '0.9', '0.5', '1.4', '7.0']
chV3_SPwaveAmp (numeric, 3 distinct): ['0.0', '-0.5', '-5.6']
chV3_PwaveAmp (numeric, 33 distinct): ['0.0', '0.3', '0.4', '0.5', '0.2', '0.6', '0.7', '0.8', '-0.3', '0.1']
chV3_TwaveAmp (numeric, 118 distinct): ['1.1', '3.1', '3.4', '3.6', '4.4', '1.0', '2.0', '4.3', '2.8', '2.6']
chV3_QRSA (numeric, 354 distinct): ['7.0', '-11.1', '-1.4', '-19.9', '-2.4', '-21.7', '-3.2', '0.2', '1.2', '-7.4']
chV3_QRSTA (numeric, 377 distinct): ['34.0', '39.5', '11.3', '8.9', '24.5', '-19.7', '43.2', '25.0', '20.1', '31.9']
chV4_JJwaveAmp (numeric, 52 distinct): ['0.1', '0.0', '-0.4', '-0.3', '-0.5', '-0.2', '-0.1', '-0.6', '0.5', '0.6']
chV4_QwaveAmp (numeric, 25 distinct): ['0.0', '-0.4', '-0.5', '-0.9', '-0.7', '-0.6', '-1.0', '-1.1', '-0.8', '-1.6']
chV4_RwaveAmp (numeric, 197 distinct): ['10.0', '12.6', '0.0', '10.9', '9.4', '7.6', '9.2', '9.7', '11.8', '8.4']
chV4_SwaveAmp (numeric, 147 distinct): ['0.0', '-4.9', '-3.5', '-7.4', '-6.1', '-3.4', '-5.6', '-6.2', '-5.5', '-4.0']
chV4_RPwaveAmp (numeric, 10 distinct): ['0.0', '0.5', '0.4', '0.8', '0.7', '2.4', '1.0', '1.4', '0.6', '0.9']
chV4_SPwaveAmp (numeric, 3 distinct): ['0.0', '-0.9', '-0.4']
chV4_PwaveAmp (numeric, 30 distinct): ['0.6', '0.5', '0.7', '0.4', '0.8', '0.3', '1.0', '0.9', '1.1', '0.2']
chV4_TwaveAmp (numeric, 103 distinct): ['2.6', '0.9', '2.5', '1.7', '3.3', '2.3', '2.8', '3.1', '0.6', '0.5']
chV4_QRSA (numeric, 336 distinct): ['6.6', '0.0', '17.7', '17.0', '9.5', '16.2', '12.6', '34.3', '3.0', '23.2']
chV4_QRSTA (numeric, 384 distinct): ['17.0', '14.2', '25.4', '49.8', '48.3', '45.9', '21.6', '39.6', '71.1', '54.4']
chV5_JJwaveAmp (numeric, 42 distinct): ['0.0', '0.1', '-0.4', '-0.5', '-0.2', '-0.1', '-0.3', '-0.6', '-0.7', '0.2']
chV5_QwaveAmp (numeric, 25 distinct): ['0.0', '-0.4', '-0.5', '-0.9', '-0.6', '-0.8', '-1.1', '-0.7', '-1.2', '-1.0']
chV5_RwaveAmp (numeric, 168 distinct): ['8.3', '10.3', '11.0', '9.3', '11.2', '13.4', '9.1', '15.9', '11.4', '12.7']
chV5_SwaveAmp (numeric, 101 distinct): ['0.0', '-2.9', '-2.5', '-1.9', '-2.3', '-3.4', '-2.6', '-2.0', '-4.3', '-3.0']
chV5_RPwaveAmp (numeric, 4 distinct): ['0.0', '0.5', '0.4', '5.8']
chV5_SPwaveAmp (numeric, 1 distinct): ['0']
chV5_PwaveAmp (numeric, 28 distinct): ['0.5', '0.4', '0.6', '0.7', '0.8', '0.9', '0.3', '1.0', '-0.1', '0.2']
chV5_TwaveAmp (numeric, 80 distinct): ['2.2', '0.6', '1.8', '1.5', '2.8', '0.5', '1.7', '1.9', '1.6', '2.0']
chV5_QRSA (numeric, 307 distinct): ['14.4', '8.5', '29.7', '-3.1', '11.7', '11.2', '33.6', '12.6', '30.1', '21.2']
chV5_QRSTA (numeric, 348 distinct): ['41.4', '33.6', '22.2', '48.5', '38.9', '29.3', '24.6', '25.1', '23.7', '18.3']
chV6_JJwaveAmp (numeric, 37 distinct): ['-0.2', '0.0', '0.1', '-0.4', '-0.1', '-0.3', '-0.5', '-0.6', '-0.7', '-0.9']
chV6_QwaveAmp (numeric, 22 distinct): ['0.0', '-0.4', '-0.6', '-0.5', '-0.7', '-0.9', '-0.8', '-1.1', '-1.3', '-1.0']
chV6_RwaveAmp (numeric, 136 distinct): ['8.5', '10.2', '8.2', '6.3', '9.5', '6.9', '8.7', '6.6', '6.8', '11.2']
chV6_SwaveAmp (numeric, 57 distinct): ['0.0', '-1.3', '-0.8', '-1.2', '-1.6', '-2.1', '-0.9', '-0.6', '-2.0', '-1.8']
chV6_RPwaveAmp (numeric, 3 distinct): ['0.0', '0.5', '0.8']
chV6_SPwaveAmp (numeric, 1 distinct): ['0']
chV6_PwaveAmp (numeric, 24 distinct): ['0.5', '0.4', '0.6', '0.7', '0.8', '0.3', '0.9', '1.0', '0.2', '-0.3']
chV6_TwaveAmp (numeric, 71 distinct): ['1.5', '1.6', '2.1', '1.7', '2.3', '1.2', '1.3', '0.9', '0.4', '0.6']
chV6_QRSA (numeric, 286 distinct): ['24.0', '17.1', '18.1', '9.4', '17.6', '26.4', '24.1', '17.9', '11.8', '14.7']
chV6_QRSTA (numeric, 332 distinct): ['16.2', '31.1', '25.7', '33.7', '19.7', '15.7', '31.7', '21.1', '12.8', '22.4']
'''

CONTEXT = "Heart Arrhythmia Detection from ECG Records"
TARGET = CuratedTarget(raw_name="class", new_name="Diagnosis", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"1": "Normal",
                                      "2": "Ischemic changes (Coronary Artery Disease)",
                                      "3": "Old Anterior Myocardial Infarction",
                                      "4": "Old Inferior Myocardial Infarction",
                                      "5": "Sinus tachycardy",
                                      "6": "Sinus bradycardy",
                                      "7": "Ventricular Premature Contraction (PVC)",
                                      "8": "Supraventricular Premature Contraction",
                                      "9": "Left bundle branch block",
                                      "10": "Right bundle branch block",
                                      # "11": "1. degree AtrioVentricular block",
                                      # "12": "2. degree AV block",
                                      # "13": "3. degree AV block",
                                      "14": "Left ventricule hypertrophy",
                                      "15": "Atrial Fibrillation or Flutter",
                                      "16": "Others"})
COLS_TO_DROP = []
FEATURES = []
