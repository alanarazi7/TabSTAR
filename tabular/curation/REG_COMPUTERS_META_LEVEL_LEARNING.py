from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: meta
====
Examples: 528
====
URL: https://www.openml.org/search?type=data&id=566
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

1. Title: meta-data

2. Sources:
(a) Creator:
LIACC - University of Porto
R.Campo Alegre 823
4150 PORTO
(b) Donor: P.B.Brazdil or J.Gama            Tel.:  +351 600 1672
LIACC, University of Porto               Fax.:  +351 600 3654
Rua Campo Alegre 823                     Email:  statlog-adm@ncc.up.pt
4150 Porto, Portugal
(c) Date: March, 1996

(d) Acknowlegements:
LIACC wishes to thank Commission of European Communities
for their support. Also, we wish to thank the following partners
for providing the individual test results:

- Dept. of Statistics, University of Strathclyde, Glasgow, UK
- Dept. of Statistics, University of Leeds, UK
- Aston University, Birmingham, UK
- Forschungszentrum Ulm, Daimler-Benz AG, Germany
- Brainware GmbH, Berlin, Germany
- Frauenhofer Gesellschaft IITB-EPO, Berlin, Germany
- Institut fuer Kybernetik, Bochum, Germany
- ISoft, Gif sur Yvette, France
- Dept. of CS and AI, University of Granada, Spain


3. Past Usage:


Meta-Data was used in order to give advice about which classification
method is appropriate for a particular dataset.
This work is described in:

-"Machine Learning, Neural and Statistical Learning"
Eds. D.Michie,D.J.Spiegelhalter and C.Taylor
Ellis Horwood-1994

- "Characterizing the Applicability of
Classification Algorithms Using Meta-Level Learning",
P. Brazdil, J.Gama and B.Henery:
in Proc. of Machine Learning - ECML-94,
ed. F.Bergadano and L.de Raedt,LNAI Vol.784 Springer-Verlag.

-"Characterization of Classification Algorithms"
J.Gama, P.Brazdil
in Proc. of EPIA 95, LNAI Vol.990
Springer-Verlag, 1995


4. Relevant Information:n
This DataSet is about the results of Statlog project.
The project performed a comparative study between Statistical, Neural
and Symbolic learning algorithms.

Project StatLog (Esprit Project 5170) was concerned with comparative
studies of different machine learning, neural and statistical
classification algorithms. About 20 different algorithms were
evaluated on more than 20 different datasets. The tests carried out
under project produced many interesting results.

Algorithms                      DataSets
-------------------------       --------------------------
C4.5            NewId           Credit_Austr    Belgian
AC2             CART            Chromosome      Credit_Man
IndCART         Cal5            CUT             DNA
CN2             ITRule          Diabetes        Digits44
Discrim         QuaDisc         Credit_German   Faults
LogDisc         ALLOC80         Head            Heart
kNN             SMART           KLDigits        Letters
BayesTree       CASTLE          New_Belgian     Sat_Image
DIPLO92         RBF             Segment         Shuttle
LVQ             Backprop        Technical       TseTse
Kohonen                         Vehicle


The results of these tests are comprehensively described in a book
(D.Michie et.al, 1994).

5. Number of Instances: 528

6. Number of Attributes: 22 (including an Id#) plus the class attribute
-- all but two attributes are continuously valued

7. Attribute Information:
1.   DS_Name         categorical     Name of DataSet
2.   T               continuous      Number of examples in test set
3.   N               continuous      Number of examples
4.   p               continuous      Number of attributes
5.   k               continuous      Number of classes
6.   Bin             continuous      Number of binary Attributes
7.   Cost            continuous      Cost (1=yes,0=no)
8.   SDratio         continuous      Standard deviation ratio
9.   correl          continuous      Mean correlation between attributes
10.   cancor1         continuous      First canonical correlation
11.   cancor2         continuous      Second canonical correlation
12.   fract1          continuous      First eigenvalue
13.   fract2          continuous      Second eigenvalue
14.   skewness        continuous      Mean of |E(X-Mean)|^3/STD^3
15.   kurtosis        continuous      Mean of |E(X-Mean)|^4/STD^4
16.   Hc              continuous      Mean entropy of attributes
17.   Hx              continuous      Entropy of classes
18.   MCx             continuous      Mean mutual entropy of class and attributes
19.   EnAtr           continuous      Equivalent number of attributes
20.   NSRatio         continuous      Noise-signal ratio
21.   Alg_Name        categorical     Name of Algorithm
22.   Norm_error      continuous      Normalized Error (continuous class)


8. Missing Attribute Values:

Note that fract2 and cancor2 only apply to datasets with more than
2 classes. When they appear as '?' this means a don't care value.

Summary Statistics:

Attribute       Min     Max     Mean    Std
T               270     20000   4569.05 5704.01
N               270     58000   10734.2 14568.8
p               6       180     29.5455 36.8533
k               2       91      9.72727 19.3568
Bin             0       43      3.18182 9.29227
Cost            0       1       0.13636 0.35125
SdRatio         1.0273  4.0014  1.4791  0.65827
Correl          0.0456  0.751   0.23684 0.1861
Cancor1         0.5044  0.9884  0.79484 0.15639
Cancor2         0.1057  0.9623  0.74106 0.269
Fract1          0.1505  1       0.70067 0.3454
Fract2          0.2807  1       0.70004 0.29405
Skewness        0.1802  6.7156  1.78422 1.79022
Kurtosis        0.9866  160.311 22.6672 41.8496
Hc              0.2893  4.8787  1.87158 1.44665
Hx              0.3672  6.5452  3.34502 1.80383
Mcx             0.0187  1.3149  0.31681 0.33548
EnAtr           1.56006 160.644 20.6641 35.6614
NsRatio         1.02314 159.644 28.873  37.925
====
Target Variable: class (numeric, 436 distinct): ['0.0', '161.514', '9.523', '12.819', '8.582', '2.711', '9.939', '467.007', '3.893', '0.779']
====
Features:

DS_Name (nominal, 22 distinct): ['Aust_Credit', 'BT', 'TseTse', 'Technical', 'Shuttle', 'Segment', 'SatImage', 'NewBelgian', 'Letters', 'KlDigits']
T (numeric, 20 distinct): ['1000.0', '9000.0', '900.0', '1499.0', '2580.0', '14500.0', '2310.0', '2000.0', '5000.0', '270.0']
N (numeric, 20 distinct): ['18000.0', '20000.0', '690.0', '900.0', '4999.0', '7078.0', '58000.0', '2310.0', '6435.0', '3000.0']
p (numeric, 18 distinct): ['16', '14', '13', '56', '9', '11', '36', '57', '40', '6']
k (numeric, 9 distinct): ['2', '3', '10', '7', '24', '26', '6', '91', '4']
Bin (numeric, 7 distinct): ['0', '4', '8', '43', '9', '1', '5']
Cost (numeric, 2 distinct): ['0', '1']
SDratio (numeric, 22 distinct): ['1.2623', '1.0975', '1.1316', '2.2442', '1.6067', '4.0014', '1.297', '1.0638', '1.8795', '1.9657']
correl (numeric, 22 distinct): ['0.1024', '0.1217', '0.3676', '0.3558', '0.1425', '0.5977', '0.1216', '0.2577', '0.1093', '0.1236']
cancor1 (numeric, 22 distinct): ['0.7713', '0.6109', '0.7792', '0.9165', '0.9668', '0.976', '0.9366', '0.5286', '0.8896', '0.9207']
cancor2 (numeric, 13 distinct): ['0.9191', '0.83', '0.8902', '0.3002', '0.1057', '0.9056', '0.8489', '0.9332', '0.9623', '0.6968']
fract1 (numeric, 13 distinct): ['1.0', '0.1505', '0.5252', '0.2031', '0.8966', '0.9787', '0.172', '0.168', '0.3586', '0.3098']
fract2 (numeric, 11 distinct): ['1.0', '0.2807', '0.4049', '0.3385', '0.321', '0.7146', '0.611', '0.9499', '0.866', '0.9139']
skewness (numeric, 22 distinct): ['1.9701', '6.1012', '0.6483', '6.7156', '4.4371', '2.958', '0.7316', '1.118', '0.5698', '0.1802']
kurtosis (numeric, 22 distinct): ['12.5538', '93.1399', '4.3322', '108.296', '160.311', '24.4813', '4.1737', '6.7738', '3.5385', '2.92']
Hc (numeric, 21 distinct): ['3.3219', '0.9912', '1.3574', '0.9998', '4.8787', '0.9653', '2.8072', '2.4734', '0.3879', '4.6996']
Hx (numeric, 22 distinct): ['2.3012', '2.7416', '3.8755', '0.3672', '3.4271', '3.0787', '5.5759', '3.83', '3.094', '5.5903']
MCx (numeric, 22 distinct): ['0.113', '0.0495', '0.285', '0.1815', '0.3348', '0.6672', '0.9443', '0.0421', '0.5189', '0.2029']
EnAtr (numeric, 22 distinct): ['8.7717', '5.8444', '3.5081', '26.8799', '2.8832', '4.2074', '2.6193', '9.2138', '9.0569', '16.3721']
NSRatio (numeric, 22 distinct): ['19.3646', '54.3859', '12.5982', '1.0231', '9.2363', '3.6144', '4.9048', '89.9739', '4.9626', '26.552']
Alg_Name (nominal, 24 distinct): ['Ac2', 'Alloc80', 'RBF', 'QuaDisc', 'NewId', 'LogDisc', 'LVQ', 'Kohonen', 'KNN', 'IndCART']
'''

CONTEXT = "Meta Level Learning"
TARGET = CuratedTarget(raw_name="class", new_name="Norm_error",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []