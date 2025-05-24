from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: MiceProtein
====
Examples: 1080
====
URL: https://www.openml.org/search?type=data&id=40966
====
Description: **Author**: Clara Higuera, Katheleen J. Gardiner, Krzysztof J. Cios  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) - 2015   
**Please cite**: Higuera C, Gardiner KJ, Cios KJ (2015) Self-Organizing Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down Syndrome. PLoS ONE 10(6): e0129126.

Expression levels of 77 proteins measured in the cerebral cortex of 8 classes of control and Down syndrome mice exposed to context fear conditioning, a task used to assess associative learning.

The data set consists of the expression levels of 77 proteins/protein modifications that produced detectable signals in the nuclear fraction of cortex. There are 38 control mice and 34 trisomic mice (Down syndrome), for a total of 72 mice. In the experiments, 15 measurements were registered of each protein per sample/mouse. Therefore, for control mice, there are 38x15, or 570 measurements, and for trisomic mice, there are 34x15, or 510 measurements. The dataset contains a total of 1080 measurements per protein. Each measurement can be considered as an independent sample/mouse. 

The eight classes of mice are described based on features such as genotype, behavior and treatment. According to genotype, mice can be control or trisomic. According to behavior, some mice have been stimulated to learn (context-shock) and others have not (shock-context) and in order to assess the effect of the drug memantine in recovering the ability to learn in trisomic mice, some mice have been injected with the drug and others have not. 

Classes: 
```
* c-CS-s: control mice, stimulated to learn, injected with saline (9 mice) 
* c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice) 
* c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice) 
* c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
* t-CS-s: trisomy mice, stimulated to learn, injected with saline (7 mice) 
* t-CS-m: trisomy mice, stimulated to learn, injected with memantine (9 mice) 
* t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice) 
* t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice) 
```

The aim is to identify subsets of proteins that are discriminant between the classes. 

### Attribute Information:

```
1 Mouse ID 
2..78 Values of expression levels of 77 proteins; the names of proteins are followed by &acirc;&euro;&oelig;_n&acirc;&euro; indicating that they were measured in the nuclear fraction. For example: DYRK1A_n 
79 Genotype: control (c) or trisomy (t) 
80 Treatment type: memantine (m) or saline (s) 
81 Behavior: context-shock (CS) or shock-context (SC) 
82 Class: c-CS-s, c-CS-m, c-SC-s, c-SC-m, t-CS-s, t-CS-m, t-SC-s, t-SC-m 
```

### Relevant Papers:

Higuera C, Gardiner KJ, Cios KJ (2015) Self-Organizing Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down Syndrome. PLoS ONE 10(6): e0129126. [Web Link] journal.pone.0129126 

Ahmed MM, Dhanasekaran AR, Block A, Tong S, Costa ACS, Stasko M, et al. (2015) Protein Dynamics Associated with Failed and Rescued Learning in the Ts65Dn Mouse Model of Down Syndrome. PLoS ONE 10(3): e0119491.
====
Target Variable: class (nominal, 8 distinct): ['c-CS-m', 'c-SC-m', 'c-CS-s', 'c-SC-s', 't-CS-m', 't-SC-m', 't-SC-s', 't-CS-s']
====
Features:

DYRK1A_N (numeric, 1078 distinct): ['0.5036', '0.3172', '0.2837', '0.2726', '0.249', '0.2553', '0.2594', '0.2266', '0.2265', '0.2274']
ITSN1_N (numeric, 1077 distinct): ['0.6393', '0.7472', '0.5112', '0.623', '0.4619', '0.4433', '0.444', '0.479', '0.4903', '0.5184']
BDNF_N (numeric, 1078 distinct): ['0.4302', '0.3195', '0.284', '0.3024', '0.2792', '0.2966', '0.2781', '0.2705', '0.2745', '0.3101']
NR1_N (numeric, 1078 distinct): ['2.8163', '2.5218', '2.3212', '2.3784', '2.2296', '2.242', '2.1649', '2.1131', '2.0876', '2.093']
NR2A_N (numeric, 1078 distinct): ['5.9902', '4.6501', '3.6851', '3.6838', '3.3904', '3.4325', '3.2937', '3.0979', '2.9062', '2.8826']
pAKT_N (numeric, 1077 distinct): ['0.2312', '0.2188', '0.2593', '0.2449', '0.2499', '0.2874', '0.2611', '0.2739', '0.2644', '0.2606']
pBRAF_N (numeric, 1076 distinct): ['0.2042', '0.2105', '0.1776', '0.1903', '0.1907', '0.2015', '0.2063', '0.1922', '0.2035', '0.1976']
pCAMKII_N (numeric, 1078 distinct): ['2.3737', '3.7106', '4.0375', '4.0007', '3.667', '3.7607', '3.6006', '3.4876', '3.4916', '3.3126']
pCREB_N (numeric, 1078 distinct): ['0.2322', '0.2507', '0.2353', '0.2421', '0.2194', '0.2323', '0.2395', '0.2338', '0.2311', '0.257']
pELK_N (numeric, 1078 distinct): ['1.7509', '1.4356', '1.1086', '1.2813', '1.1348', '1.0857', '1.1895', '1.1145', '1.1068', '1.2109']
pERK_N (numeric, 1078 distinct): ['0.6879', '0.347', '0.3213', '0.3129', '0.2832', '0.2852', '0.3038', '0.2758', '0.2733', '0.2562']
pJNK_N (numeric, 1077 distinct): ['0.3333', '0.347', '0.3453', '0.3252', '0.3293', '0.3142', '0.3203', '0.3171', '0.329', '0.3173']
PKCA_N (numeric, 1078 distinct): ['0.4027', '0.334', '0.2865', '0.319', '0.2772', '0.271', '0.287', '0.277', '0.2697', '0.2818']
pMEK_N (numeric, 1078 distinct): ['0.2969', '0.2994', '0.2789', '0.2894', '0.281', '0.2776', '0.2943', '0.3117', '0.2939', '0.2914']
pNR1_N (numeric, 1078 distinct): ['1.0221', '0.88', '0.8477', '0.8892', '0.8321', '0.8502', '0.8208', '0.7847', '0.7741', '0.7839']
pNR2A_N (numeric, 1078 distinct): ['0.6057', '0.9879', '0.8903', '0.8712', '0.8089', '0.82', '0.7941', '0.7545', '0.7431', '0.7238']
pNR2B_N (numeric, 1078 distinct): ['1.8777', '1.767', '1.6803', '1.6141', '1.5288', '1.5769', '1.4774', '1.4496', '1.4368', '1.3529']
pPKCAB_N (numeric, 1078 distinct): ['2.3087', '1.1475', '0.9562', '0.9729', '0.9519', '0.9443', '0.8928', '0.9519', '0.9262', '0.9823']
pRSK_N (numeric, 1078 distinct): ['0.4416', '0.4721', '0.4885', '0.4686', '0.4603', '0.489', '0.45', '0.4685', '0.4882', '0.4903']
AKT_N (numeric, 1078 distinct): ['0.8594', '0.829', '0.6468', '0.7082', '0.6277', '0.6441', '0.686', '0.5948', '0.6036', '0.6793']
BRAF_N (numeric, 1078 distinct): ['0.4163', '0.228', '0.2486', '0.2702', '0.2801', '0.2559', '0.2762', '0.2734', '0.2639', '0.3416']
CAMKII_N (numeric, 1078 distinct): ['0.3696', '0.3665', '0.3688', '0.3738', '0.3472', '0.3541', '0.381', '0.3808', '0.3707', '0.378']
CREB_N (numeric, 1074 distinct): ['0.1979', '0.2289', '0.186', '0.2109', '0.1789', '0.1915', '0.1831', '0.1778', '0.2078', '0.1891']
ELK_N (numeric, 1063 distinct): ['1.8664', '1.1201', '1.0817', '1.0144', '0.981', '1.062', '0.9711', '0.9707', '1.6073', '1.5603']
ERK_N (numeric, 1078 distinct): ['3.6852', '3.0165', '2.4974', '2.4258', '2.3173', '2.2626', '2.1188', '2.0918', '2.0656', '1.9943']
GSK3B_N (numeric, 1078 distinct): ['1.5372', '1.1714', '1.0224', '1.026', '0.9775', '0.9402', '0.9743', '0.9364', '0.9268', '0.8949']
JNK_N (numeric, 1078 distinct): ['0.2645', '0.2709', '0.2315', '0.237', '0.2289', '0.2255', '0.2377', '0.2303', '0.2215', '0.2284']
MEK_N (numeric, 1073 distinct): ['0.2752', '0.2485', '0.297', '0.3135', '0.327', '0.267', '0.2396', '0.2483', '0.3197', '0.262']
TRKA_N (numeric, 1076 distinct): ['0.6667', '0.7769', '0.7418', '0.6696', '0.7051', '0.7218', '0.6876', '0.6979', '0.6521', '0.6513']
RSK_N (numeric, 1075 distinct): ['0.1877', '0.1956', '0.1622', '0.201', '0.1585', '0.2098', '0.2041', '0.2156', '0.1847', '0.1958']
APP_N (numeric, 1078 distinct): ['0.4539', '0.51', '0.3886', '0.401', '0.3647', '0.3638', '0.3698', '0.3546', '0.3633', '0.3465']
Bcatenin_N (numeric, 1063 distinct): ['3.0376', '2.3645', '1.989', '1.9196', '1.9442', '1.7657', '1.7158', '1.7992', '3.2434', '3.1732']
SOD1_N (numeric, 1078 distinct): ['0.3695', '1.592', '1.1613', '1.0752', '0.9931', '0.9939', '0.9959', '0.9748', '0.947', '1.0751']
MTOR_N (numeric, 1078 distinct): ['0.4585', '0.4683', '0.391', '0.3948', '0.3775', '0.3801', '0.4052', '0.4033', '0.3976', '0.4056']
P38_N (numeric, 1076 distinct): ['0.3333', '0.4729', '0.371', '0.4666', '0.3568', '0.3593', '0.4009', '0.3541', '0.3667', '0.592']
pMTOR_N (numeric, 1078 distinct): ['0.8252', '0.9002', '0.8126', '0.7698', '0.7252', '0.7485', '0.7401', '0.7585', '0.7491', '0.7615']
DSCR1_N (numeric, 1078 distinct): ['0.5769', '0.6583', '0.5544', '0.5881', '0.5479', '0.5644', '0.591', '0.5731', '0.5732', '0.6032']
AMPKA_N (numeric, 1076 distinct): ['0.3333', '0.3677', '0.4481', '0.3065', '0.3994', '0.4244', '0.3246', '0.3062', '0.3067', '0.3142']
NR2B_N (numeric, 1078 distinct): ['0.5863', '0.6013', '0.5402', '0.5332', '0.5482', '0.474', '0.5182', '0.5297', '0.5308', '0.5292']
pNUMB_N (numeric, 1078 distinct): ['0.3947', '0.3473', '0.3227', '0.3281', '0.3014', '0.297', '0.336', '0.3056', '0.2943', '0.3365']
RAPTOR_N (numeric, 1078 distinct): ['0.3396', '0.3904', '0.3', '0.3251', '0.3058', '0.2906', '0.3202', '0.3186', '0.3001', '0.3337']
TIAM1_N (numeric, 1076 distinct): ['0.3156', '0.375', '0.4829', '0.5071', '0.3789', '0.3667', '0.3761', '0.3617', '0.38', '0.4783']
pP70S6_N (numeric, 1077 distinct): ['0.5', '0.344', '0.368', '0.3928', '0.35', '0.336', '0.3321', '0.3355', '0.2942', '0.324']
NUMB_N (numeric, 1080 distinct): ['0.1822', '0.1847', '0.1753', '0.1763', '0.1723', '0.1509', '0.1676', '0.1765', '0.1757', '0.1781']
P70S6_N (numeric, 1080 distinct): ['0.8427', '1.1441', '0.9617', '0.9525', '0.9349', '0.8341', '0.8782', '0.8732', '0.8732', '0.8954']
pGSK3B_N (numeric, 1080 distinct): ['0.1926', '0.1639', '0.1634', '0.1793', '0.1565', '0.1617', '0.1717', '0.1741', '0.1785', '0.1798']
pPKCG_N (numeric, 1080 distinct): ['1.4431', '1.3786', '1.5763', '1.6186', '1.4905', '1.4985', '1.573', '1.5647', '1.6504', '1.6632']
CDK5_N (numeric, 1080 distinct): ['0.2947', '0.2983', '0.2597', '0.2725', '0.2695', '0.248', '0.2574', '0.2595', '0.2617', '0.2339']
S6_N (numeric, 1080 distinct): ['0.3546', '0.3495', '0.5015', '0.5292', '0.5143', '0.5201', '0.5962', '0.5535', '0.5164', '0.5283']
ADARB1_N (numeric, 1080 distinct): ['1.3391', '1.6055', '1.178', '1.1771', '1.0765', '1.1328', '1.1598', '1.1157', '1.1749', '1.1579']
AcetylH3K9_N (numeric, 1080 distinct): ['0.1701', '0.1507', '0.8289', '0.8593', '0.7699', '0.894', '1.0037', '0.8499', '0.9372', '0.932']
RRP1_N (numeric, 1080 distinct): ['0.1591', '0.1792', '0.1658', '0.1728', '0.1618', '0.1843', '0.1888', '0.1907', '0.2158', '0.2192']
BAX_N (numeric, 1080 distinct): ['0.1889', '0.1964', '0.1774', '0.1874', '0.1642', '0.1758', '0.1832', '0.1774', '0.1738', '0.1877']
ARC_N (numeric, 1080 distinct): ['0.1063', '0.1305', '0.1256', '0.1352', '0.1298', '0.1269', '0.138', '0.1383', '0.1308', '0.1397']
ERBB4_N (numeric, 1079 distinct): ['0.1553', '0.145', '0.1702', '0.1637', '0.1721', '0.1681', '0.1627', '0.1938', '0.1905', '0.1535']
nNOS_N (numeric, 1079 distinct): ['0.1739', '0.1767', '0.1932', '0.1953', '0.2057', '0.1901', '0.1783', '0.1905', '0.1839', '0.1723']
Tau_N (numeric, 1080 distinct): ['0.1252', '0.1499', '0.385', '0.3921', '0.3617', '0.3947', '0.4095', '0.4189', '0.4681', '0.5036']
GFAP_N (numeric, 1079 distinct): ['0.1304', '0.1291', '0.1232', '0.1109', '0.1113', '0.1162', '0.1314', '0.1308', '0.1153', '0.12']
GluR3_N (numeric, 1080 distinct): ['0.228', '0.2675', '0.2006', '0.1951', '0.1881', '0.1875', '0.1932', '0.1884', '0.1845', '0.1896']
GluR4_N (numeric, 1079 distinct): ['0.111', '0.1428', '0.1031', '0.1594', '0.1639', '0.1549', '0.1545', '0.1566', '0.1191', '0.1609']
IL1B_N (numeric, 1080 distinct): ['0.431', '0.5884', '0.6084', '0.6238', '0.6089', '0.5794', '0.6628', '0.6483', '0.6742', '0.6922']
P3525_N (numeric, 1080 distinct): ['0.2475', '0.2676', '0.2683', '0.2861', '0.278', '0.271', '0.3088', '0.2965', '0.3057', '0.3078']
pCASP9_N (numeric, 1080 distinct): ['1.6033', '2.0862', '1.7422', '1.8108', '1.7364', '1.7232', '1.8199', '1.8397', '1.7274', '1.7306']
PSD95_N (numeric, 1080 distinct): ['2.0149', '2.4578', '2.382', '2.4667', '2.3802', '2.2756', '2.4226', '2.5034', '2.3351', '2.4103']
SNCA_N (numeric, 1079 distinct): ['0.1715', '0.1082', '0.1762', '0.1462', '0.1461', '0.1485', '0.136', '0.175', '0.1915', '0.1585']
Ubiquitin_N (numeric, 1080 distinct): ['1.045', '1.286', '1.3233', '1.3703', '1.295', '1.3079', '1.4101', '1.45', '1.3836', '1.412']
pGSK3B_Tyr216_N (numeric, 1080 distinct): ['0.8316', '0.8807', '0.7508', '0.8069', '0.7622', '0.7459', '0.8064', '0.8428', '0.8177', '0.8669']
SHH_N (numeric, 1080 distinct): ['0.1889', '0.1992', '0.2', '0.204', '0.2065', '0.2079', '0.2175', '0.2198', '0.2128', '0.2037']
BAD_N (numeric, 867 distinct): ['0.2003', '0.1227', '0.137', '0.1378', '0.1557', '0.1496', '0.1511', '0.1468', '0.1729', '0.1507']
BCL2_N (numeric, 796 distinct): ['0.1786', '0.1635', '0.1632', '0.0954', '0.1002', '0.1003', '0.104', '0.1149', '0.1112', '0.1173']
pS6_N (numeric, 1080 distinct): ['0.1063', '0.1305', '0.1256', '0.1352', '0.1298', '0.1269', '0.138', '0.1383', '0.1308', '0.1397']
pCFOS_N (numeric, 1006 distinct): ['0.1083', '0.1256', '0.0988', '0.1005', '0.1019', '0.1027', '0.1133', '0.1051', '0.1048', '0.1151']
SYP_N (numeric, 1079 distinct): ['0.488', '0.4271', '0.4855', '0.4824', '0.514', '0.49', '0.4961', '0.4991', '0.4681', '0.5326']
H3AcK18_N (numeric, 901 distinct): ['0.1148', '0.1254', '0.4097', '0.4295', '0.4602', '0.4536', '0.4551', '0.4798', '0.4356', '0.2488']
EGR1_N (numeric, 871 distinct): ['0.1318', '0.3145', '0.2185', '0.2106', '0.2135', '0.2095', '0.2301', '0.2232', '0.2844', '0.2596']
H3MeK4_N (numeric, 811 distinct): ['0.1282', '0.1834', '0.2206', '0.2348', '0.2385', '0.2424', '0.2906', '0.2952', '0.2907', '0.1665']
CaNA_N (numeric, 1080 distinct): ['1.6757', '1.2066', '1.2701', '1.306', '1.2127', '1.2261', '1.3011', '1.2548', '1.2694', '1.2997']
'''

CONTEXT = "Mice Protein Expression Dataset"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []