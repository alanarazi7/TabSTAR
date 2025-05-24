from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: qsar-biodeg
====
Examples: 1055
====
URL: https://www.openml.org/search?type=data&id=1494
====
Description: **Author**: Kamel Mansouri, Tine Ringsted, Davide Ballabio  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation)  
**Please cite**: Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878 


QSAR biodegradation Data Set 

* Abstract: 

Data set containing values for 41 attributes (molecular descriptors) used to classify 1055 chemicals into 2 classes (ready and not ready biodegradable).


* Source:

Kamel Mansouri, Tine Ringsted, Davide Ballabio (davide.ballabio '@' unimib.it), Roberto Todeschini, Viviana Consonni, Milano Chemometrics and QSAR Research Group (http://michem.disat.unimib.it/chm/), UniversitÃ  degli Studi Milano â€“ Bicocca, Milano (Italy)


* Data Set Information:

The QSAR biodegradation dataset was built in the Milano Chemometrics and QSAR Research Group (UniversitÃ  degli Studi Milano â€“ Bicocca, Milano, Italy). The research leading to these results has received funding from the European Communityâ€™s Seventh Framework Programme [FP7/2007-2013] under Grant Agreement n. 238701 of Marie Curie ITN Environmental Chemoinformatics (ECO) project. 
The data have been used to develop QSAR (Quantitative Structure Activity Relationships) models for the study of the relationships between chemical structure and biodegradation of molecules. Biodegradation experimental values of 1055 chemicals were collected from the webpage of the National Institute of Technology and Evaluation of Japan (NITE). Classification models were developed in order to discriminate ready (356) and not ready (699) biodegradable molecules by means of three different modelling methods: k Nearest Neighbours, Partial Least Squares Discriminant Analysis and Support Vector Machines. Details on attributes (molecular descriptors) selected in each model can be found in the quoted reference: Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878.


* Attribute Information:

41 molecular descriptors and 1 experimental class: 
1) SpMax_L: Leading eigenvalue from Laplace matrix 
2) J_Dz(e): Balaban-like index from Barysz matrix weighted by Sanderson electronegativity 
3) nHM: Number of heavy atoms 
4) F01[N-N]: Frequency of N-N at topological distance 1 
5) F04[C-N]: Frequency of C-N at topological distance 4 
6) NssssC: Number of atoms of type ssssC 
7) nCb-: Number of substituted benzene C(sp2) 
8) C%: Percentage of C atoms 
9) nCp: Number of terminal primary C(sp3) 
10) nO: Number of oxygen atoms 
11) F03[C-N]: Frequency of C-N at topological distance 3 
12) SdssC: Sum of dssC E-states 
13) HyWi_B(m): Hyper-Wiener-like index (log function) from Burden matrix weighted by mass 
14) LOC: Lopping centric index 
15) SM6_L: Spectral moment of order 6 from Laplace matrix 
16) F03[C-O]: Frequency of C - O at topological distance 3 
17) Me: Mean atomic Sanderson electronegativity (scaled on Carbon atom) 
18) Mi: Mean first ionization potential (scaled on Carbon atom) 
19) nN-N: Number of N hydrazines 
20) nArNO2: Number of nitro groups (aromatic) 
21) nCRX3: Number of CRX3 
22) SpPosA_B(p): Normalized spectral positive sum from Burden matrix weighted by polarizability 
23) nCIR: Number of circuits 
24) B01[C-Br]: Presence/absence of C - Br at topological distance 1 
25) B03[C-Cl]: Presence/absence of C - Cl at topological distance 3 
26) N-073: Ar2NH / Ar3N / Ar2N-Al / R..N..R 
27) SpMax_A: Leading eigenvalue from adjacency matrix (Lovasz-Pelikan index) 
28) Psi_i_1d: Intrinsic state pseudoconnectivity index - type 1d 
29) B04[C-Br]: Presence/absence of C - Br at topological distance 4 
30) SdO: Sum of dO E-states 
31) TI2_L: Second Mohar index from Laplace matrix 
32) nCrt: Number of ring tertiary C(sp3) 
33) C-026: R--CX--R 
34) F02[C-N]: Frequency of C - N at topological distance 2 
35) nHDon: Number of donor atoms for H-bonds (N and O) 
36) SpMax_B(m): Leading eigenvalue from Burden matrix weighted by mass 
37) Psi_i_A: Intrinsic state pseudoconnectivity index - type S average 
38) nN: Number of Nitrogen atoms 
39) SM6_B(m): Spectral moment of order 6 from Burden matrix weighted by mass 
40) nArCOOR: Number of esters (aromatic) 
41) nX: Number of halogen atoms 
42) experimental class: ready biodegradable (RB) and not ready biodegradable (NRB)


* Relevant Papers:

Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 440 distinct): ['4.414', '4.732', '4.0', '4.17', '4.562', '4.807', '4.303', '4.77', '4.499', '3.618']
V2 (numeric, 1022 distinct): ['2.4328', '2.4062', '3.0864', '3.1356', '3.3983', '4.2631', '3.0402', '2.7059', '1.8789', '3.1116']
V3 (numeric, 11 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '10', '7']
V4 (numeric, 4 distinct): ['0', '1', '2', '3']
V5 (numeric, 16 distinct): ['0', '2', '1', '3', '4', '6', '7', '9', '8', '5']
V6 (numeric, 13 distinct): ['0', '1', '2', '3', '4', '6', '9', '5', '8', '11']
V7 (numeric, 15 distinct): ['0', '2', '3', '4', '1', '6', '5', '8', '7', '9']
V8 (numeric, 188 distinct): ['33.3', '40.0', '50.0', '25.0', '42.9', '28.6', '30.0', '37.5', '46.2', '41.2']
V9 (numeric, 15 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '9', '7']
V10 (numeric, 12 distinct): ['0', '2', '1', '4', '3', '6', '5', '7', '8', '12']
V11 (numeric, 21 distinct): ['0', '2', '4', '1', '3', '6', '8', '5', '12', '10']
V12 (numeric, 384 distinct): ['0.0', '-0.117', '0.134', '-1.072', '-0.185', '-0.98', '0.642', '-0.664', '-1.093', '-0.888']
V13 (numeric, 756 distinct): ['3.647', '3.66', '3.375', '3.192', '3.462', '3.233', '3.37', '2.938', '3.699', '4.049']
V14 (numeric, 373 distinct): ['0.0', '0.875', '1.185', '0.881', '1.16', '1.187', '0.971', '1.459', '0.918', '0.802']
V15 (numeric, 510 distinct): ['9.54', '8.597', '9.882', '9.833', '9.183', '10.099', '9.311', '8.755', '9.863', '7.408']
V16 (numeric, 24 distinct): ['0', '2', '4', '6', '8', '1', '3', '12', '9', '10']
V17 (numeric, 167 distinct): ['0.998', '0.979', '0.993', '0.974', '0.987', '0.991', '1.014', '0.98', '0.983', '1.011']
V18 (numeric, 125 distinct): ['1.127', '1.139', '1.125', '1.14', '1.129', '1.146', '1.121', '1.141', '1.132', '1.144']
V19 (numeric, 3 distinct): ['0', '1', '2']
V20 (numeric, 4 distinct): ['0', '1', '2', '3']
V21 (numeric, 4 distinct): ['0', '1', '2', '3']
V22 (numeric, 352 distinct): ['1.195', '1.299', '1.254', '1.296', '1.253', '1.28', '1.215', '1.211', '1.202', '1.295']
V23 (numeric, 13 distinct): ['1', '0', '2', '3', '6', '4', '7', '15', '5', '10']
V24 (numeric, 2 distinct): ['0', '1']
V25 (numeric, 2 distinct): ['0', '1']
V26 (numeric, 4 distinct): ['0', '1', '2', '3']
V27 (numeric, 329 distinct): ['2.0', '2.236', '2.194', '1.848', '2.175', '2.303', '2.101', '1.732', '1.902', '2.136']
V28 (numeric, 205 distinct): ['0.0', '-0.008', '0.001', '0.004', '-0.002', '-0.001', '0.014', '-0.025', '0.015', '-0.007']
V29 (numeric, 2 distinct): ['0', '1']
V30 (numeric, 470 distinct): ['0.0', '9.431', '22.204', '10.87', '11.013', '10.319', '20.772', '10.696', '19.107', '10.337']
V31 (numeric, 553 distinct): ['1.542', '1.06', '0.975', '0.95', '1.74', '1.481', '1.0', '2.052', '1.14', '1.707']
V32 (numeric, 8 distinct): ['0', '1', '2', '4', '6', '3', '5', '8']
V33 (numeric, 11 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '12', '10']
V34 (numeric, 16 distinct): ['0', '2', '4', '3', '1', '6', '8', '5', '10', '18']
V35 (numeric, 8 distinct): ['0', '1', '2', '3', '4', '6', '5', '7']
V36 (numeric, 705 distinct): ['3.309', '6.88', '4.009', '3.423', '3.497', '3.718', '3.882', '3.627', '3.648', '3.712']
V37 (numeric, 624 distinct): ['2.833', '2.167', '2.5', '2.667', '1.833', '2.0', '2.333', '2.802', '2.25', '3.133']
V38 (numeric, 8 distinct): ['0', '1', '2', '3', '4', '6', '5', '8']
V39 (numeric, 862 distinct): ['8.143', '8.015', '8.704', '8.506', '8.601', '8.497', '8.68', '8.128', '9.118', '8.562']
V40 (numeric, 5 distinct): ['0', '2', '1', '4', '3']
V41 (numeric, 17 distinct): ['0', '1', '2', '3', '4', '6', '5', '10', '8', '27']
'''

CONTEXT = "Quantitative Structure Activity Relationships (QSAR) Biodegradation"
TARGET = CuratedTarget(raw_name="Class", new_name="Biodegradable Molecules", task_type=SupervisedTask.BINARY,
                       label_mapping={'1': 'Not Ready', '2': 'Ready'})
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name="V1", new_name="Leading eigenvalue from Laplace matrix"),
            CuratedFeature(raw_name="V2", new_name="Balaban-like index from Barysz matrix weighted by Sanderson electronegativity"),
            CuratedFeature(raw_name="V3", new_name="Number of heavy atoms"),
            CuratedFeature(raw_name="V4", new_name="Frequency of N-N at topological distance 1"),
            CuratedFeature(raw_name="V5", new_name="Frequency of C-N at topological distance 4"),
            CuratedFeature(raw_name="V6", new_name="Number of atoms of type ssssC"),
            CuratedFeature(raw_name="V7", new_name="Number of substituted benzene C(sp2)"),
            CuratedFeature(raw_name="V8", new_name="Percentage of C atoms"),
            CuratedFeature(raw_name="V9", new_name="Number of terminal primary C(sp3)"),
            CuratedFeature(raw_name="V10", new_name="Number of oxygen atoms"),
            CuratedFeature(raw_name="V11", new_name="Frequency of C-N at topological distance 3"),
            CuratedFeature(raw_name="V12", new_name="Sum of dssC E-states"),
            CuratedFeature(raw_name="V13", new_name="Hyper-Wiener-like index (log function) from Burden matrix weighted by mass"),
            CuratedFeature(raw_name="V14", new_name="Lopping centric index"),
            CuratedFeature(raw_name="V15", new_name="Spectral moment of order 6 from Laplace matrix"),
            CuratedFeature(raw_name="V16", new_name="Frequency of C - O at topological distance 3"),
            CuratedFeature(raw_name="V17", new_name="Mean atomic Sanderson electronegativity (scaled on Carbon atom)"),
            CuratedFeature(raw_name="V18", new_name="Mean first ionization potential (scaled on Carbon atom)"),
            CuratedFeature(raw_name="V19", new_name="Number of N hydrazines"),
            CuratedFeature(raw_name="V20", new_name="Number of nitro groups (aromatic)"),
            CuratedFeature(raw_name="V21", new_name="Number of CRX3"),
            CuratedFeature(raw_name="V22", new_name="Normalized spectral positive sum from Burden matrix weighted by polarizability"),
            CuratedFeature(raw_name="V23", new_name="Number of circuits"),
            CuratedFeature(raw_name="V24", new_name="Presence/absence of C - Br at topological distance 1"),
            CuratedFeature(raw_name="V25", new_name="Presence/absence of C - Cl at topological distance 3"),
            CuratedFeature(raw_name="V26", new_name="Ar2NH / Ar3N / Ar2N-Al / R..N..R"),
            CuratedFeature(raw_name="V27", new_name="Leading eigenvalue from adjacency matrix (Lovasz-Pelikan index)"),
            CuratedFeature(raw_name="V28", new_name="Intrinsic state pseudoconnectivity index - type 1d"),
            CuratedFeature(raw_name="V29", new_name="Presence/absence of C - Br at topological distance 4"),
            CuratedFeature(raw_name="V30", new_name="Sum of dO E-states"),
            CuratedFeature(raw_name="V31", new_name="Second Mohar index from Laplace matrix"),
            CuratedFeature(raw_name="V32", new_name="Number of ring tertiary C(sp3)"),
            CuratedFeature(raw_name="V33", new_name="R--CX--R"),
            CuratedFeature(raw_name="V34", new_name="Frequency of C - N at topological distance 2"),
            CuratedFeature(raw_name="V35", new_name="Number of donor atoms for H-bonds (N and O)"),
            CuratedFeature(raw_name="V36", new_name="Leading eigenvalue from Burden matrix weighted by mass"),
            CuratedFeature(raw_name="V37", new_name="Intrinsic state pseudoconnectivity index - type S average"),
            CuratedFeature(raw_name="V38", new_name="Number of Nitrogen atoms"),
            CuratedFeature(raw_name="V39", new_name="Spectral moment of order 6 from Burden matrix weighted by mass"),
            CuratedFeature(raw_name="V40", new_name="Number of esters (aromatic)"),
            CuratedFeature(raw_name="V41", new_name="Number of halogen atoms"),
]
