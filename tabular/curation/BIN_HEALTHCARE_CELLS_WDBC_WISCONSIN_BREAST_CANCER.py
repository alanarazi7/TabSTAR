from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wdbc
====
Examples: 569
====
URL: https://www.openml.org/search?type=data&id=1510
====
Description: **Author**: William H. Wolberg, W. Nick Street, Olvi L. Mangasarian    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)), [University of Wisconsin](http://pages.cs.wisc.edu/~olvi/uwmp/cancer.html) - 1995  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)     

**Breast Cancer Wisconsin (Diagnostic) Data Set (WDBC).** Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The target feature records the prognosis (benign (1) or malignant (2)). [Original data available here](ftp://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/) 

Current dataset was adapted to ARFF format from the UCI version. Sample code ID's were removed.  

! Note that there is also a related Breast Cancer Wisconsin (Original) Data Set with a different set of features, better known as [breast-w](https://www.openml.org/d/15).


### Feature description  

Ten real-valued features are computed for each of 3 cell nuclei, yielding a total of 30 descriptive features. See the papers below for more details on how they were computed. The 10 features (in order) are:  

a) radius (mean of distances from center to points on the perimeter)  
b) texture (standard deviation of gray-scale values)  
c) perimeter  
d) area  
e) smoothness (local variation in radius lengths)  
f) compactness (perimeter^2 / area - 1.0)  
g) concavity (severity of concave portions of the contour)  
h) concave points (number of concave portions of the contour)  
i) symmetry  
j) fractal dimension ("coastline approximation" - 1)  

### Relevant Papers   

W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993. 

O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 456 distinct): ['12.34', '11.71', '12.46', '13.05', '10.26', '13.85', '12.77', '13.17', '13.0', '15.46']
V2 (numeric, 479 distinct): ['20.52', '16.85', '16.84', '19.83', '14.93', '17.46', '18.9', '15.7', '18.22', '20.22']
V3 (numeric, 522 distinct): ['82.61', '87.76', '134.7', '93.97', '82.69', '120.2', '107.1', '79.19', '114.2', '58.79']
V4 (numeric, 539 distinct): ['512.2', '1075.0', '582.7', '399.8', '641.2', '394.1', '372.7', '477.3', '758.6', '1138.0']
V5 (numeric, 474 distinct): ['0.1007', '0.115', '0.1054', '0.1075', '0.1063', '0.117', '0.1049', '0.1044', '0.1066', '0.1158']
V6 (numeric, 537 distinct): ['0.1147', '0.1206', '0.077', '0.0574', '0.0383', '0.1516', '0.1117', '0.1111', '0.2087', '0.1047']
V7 (numeric, 537 distinct): ['0.0', '0.1204', '0.1115', '0.0334', '0.1103', '0.1085', '0.101', '0.0197', '0.03', '0.1007']
V8 (numeric, 542 distinct): ['0.0', '0.0286', '0.1471', '0.0578', '0.0227', '0.0237', '0.0238', '0.0259', '0.0525', '0.0203']
V9 (numeric, 432 distinct): ['0.1714', '0.1769', '0.1893', '0.1601', '0.1717', '0.1861', '0.1966', '0.1925', '0.1506', '0.1739']
V10 (numeric, 499 distinct): ['0.0611', '0.0591', '0.0591', '0.0567', '0.0678', '0.0587', '0.0602', '0.0567', '0.0641', '0.0602']
V11 (numeric, 540 distinct): ['0.286', '0.2204', '0.2684', '0.2239', '0.1601', '0.2957', '0.2562', '0.163', '0.403', '0.306']
V12 (numeric, 519 distinct): ['1.15', '1.35', '1.268', '0.8561', '1.016', '1.194', '1.095', '0.9209', '1.041', '1.216']
V13 (numeric, 533 distinct): ['1.778', '1.243', '2.569', '2.183', '3.008', '1.101', '1.959', '1.566', '2.041', '2.747']
V14 (numeric, 528 distinct): ['16.97', '17.67', '16.64', '18.54', '20.98', '20.67', '11.28', '15.75', '18.15', '24.19']
V15 (numeric, 547 distinct): ['0.0064', '0.0105', '0.0129', '0.0072', '0.0053', '0.0104', '0.0076', '0.0078', '0.0051', '0.006']
V16 (numeric, 541 distinct): ['0.0181', '0.0231', '0.011', '0.0243', '0.012', '0.0306', '0.0118', '0.014', '0.0165', '0.0222']
V17 (numeric, 533 distinct): ['0.0', '0.0165', '0.017', '0.0268', '0.0358', '0.0345', '0.0295', '0.0151', '0.0266', '0.0131']
V18 (numeric, 507 distinct): ['0.0', '0.0117', '0.0111', '0.015', '0.01', '0.0107', '0.0055', '0.0101', '0.012', '0.0115']
V19 (numeric, 498 distinct): ['0.0134', '0.0204', '0.0188', '0.0165', '0.0187', '0.019', '0.0145', '0.0154', '0.0192', '0.0126']
V20 (numeric, 545 distinct): ['0.004', '0.0032', '0.0046', '0.0057', '0.0019', '0.0027', '0.0033', '0.002', '0.0026', '0.0028']
V21 (numeric, 457 distinct): ['12.36', '13.34', '13.5', '12.84', '15.14', '13.75', '13.35', '15.53', '16.76', '19.85']
V22 (numeric, 511 distinct): ['27.26', '17.7', '16.93', '30.5', '23.17', '19.35', '26.93', '25.8', '26.44', '25.84']
V23 (numeric, 514 distinct): ['117.7', '105.9', '101.7', '184.6', '106.4', '79.93', '92.04', '145.4', '127.1', '89.0']
V24 (numeric, 544 distinct): ['472.4', '1210.0', '826.4', '402.8', '1750.0', '706.0', '830.5', '546.7', '698.8', '1269.0']
V25 (numeric, 411 distinct): ['0.1347', '0.1275', '0.1223', '0.1401', '0.1234', '0.1415', '0.1256', '0.1312', '0.1216', '0.1426']
V26 (numeric, 529 distinct): ['0.3416', '0.1486', '0.217', '0.0987', '0.2264', '0.4061', '0.1352', '0.1788', '0.1202', '0.1049']
V27 (numeric, 539 distinct): ['0.0', '0.4504', '0.1377', '0.1804', '0.1811', '0.1791', '0.363', '0.2644', '0.1423', '0.3853']
V28 (numeric, 492 distinct): ['0.0', '0.0556', '0.063', '0.1218', '0.0743', '0.1708', '0.1105', '0.0256', '0.0431', '0.1827']
V29 (numeric, 500 distinct): ['0.2226', '0.2369', '0.2972', '0.3196', '0.3109', '0.2383', '0.2268', '0.3103', '0.2576', '0.2849']
V30 (numeric, 535 distinct): ['0.0743', '0.0903', '0.1297', '0.0817', '0.1055', '0.0849', '0.087', '0.0863', '0.1019', '0.0801']
'''

CELL_FEATURE_NAMES = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity",
                      "Concave Points", "Symmetry", "Fractal Dimension"]
ALL_FEATURE_NAMES = [f"Cell {i}: {name}" for i in range(1, 4) for name in CELL_FEATURE_NAMES]
assert len(ALL_FEATURE_NAMES) == 30

CONTEXT = "Wisconsin Cells Dataset for Breast Cancer"
TARGET = CuratedTarget(raw_name="Class", new_name="Cancer Type", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "benign", "2": "malignant"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=f"V{i+1}", new_name=n) for i, n in enumerate(ALL_FEATURE_NAMES)]