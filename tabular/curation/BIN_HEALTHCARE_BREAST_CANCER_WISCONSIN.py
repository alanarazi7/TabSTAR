from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: breast-w
====
URL: https://www.openml.org/search?type=data&id=15
====
Description: **Author**: Dr. William H. Wolberg, University of Wisconsin  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)), [University of Wisconsin](http://pages.cs.wisc.edu/~olvi/uwmp/cancer.html) - 1995  
**Please cite**: See below, plus [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  

**Breast Cancer Wisconsin (Original) Data Set.** Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The target feature records the prognosis (malignant or benign). [Original data available here](ftp://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/)  

Current dataset was adapted to ARFF format from the UCI version. Sample code ID's were removed.  

! Note that there is also a related Breast Cancer Wisconsin (Diagnosis) Data Set with a different set of features, better known as [wdbc](https://www.openml.org/d/1510).

### Relevant Papers  

W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993. 

O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.  

### Citation request  

This breast cancer database was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg.  If you publish results when using this database, then please include this information in your acknowledgments.  Also, please cite one or more of:

   1. O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear 
      programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.

   2. William H. Wolberg and O.L. Mangasarian: "Multisurface method of 
      pattern separation for medical diagnosis applied to breast cytology", 
      Proceedings of the National Academy of Sciences, U.S.A., Volume 87, 
      December 1990, pp 9193-9196.

   3. O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition 
      via linear programming: Theory and application to medical diagnosis", 
      in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying
      Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.

   4. K. P. Bennett & O. L. Mangasarian: "Robust linear programming 
      discrimination of two linearly inseparable sets", Optimization Methods
      and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).
====
Target Variable: Class (nominal, 2 distinct): ['benign', 'malignant']
====
Features:

Clump_Thickness (numeric, 10 distinct): ['1.0', '5.0', '3.0', '4.0', '10.0', '2.0', '8.0', '6.0', '7.0', '9.0']
Cell_Size_Uniformity (numeric, 10 distinct): ['1.0', '10.0', '3.0', '2.0', '4.0', '5.0', '8.0', '6.0', '7.0', '9.0']
Cell_Shape_Uniformity (numeric, 10 distinct): ['1.0', '2.0', '10.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
Marginal_Adhesion (numeric, 10 distinct): ['1.0', '3.0', '2.0', '10.0', '4.0', '8.0', '5.0', '6.0', '7.0', '9.0']
Single_Epi_Cell_Size (numeric, 10 distinct): ['2.0', '3.0', '4.0', '1.0', '6.0', '5.0', '10.0', '8.0', '7.0', '9.0']
Bare_Nuclei (numeric, 26 distinct): ['1.0', '10.0', '2.0', '5.0', '3.0', '8.0', '4.0', '9.0', '7.0', '6.0']
Bland_Chromatin (numeric, 10 distinct): ['2.0', '3.0', '1.0', '7.0', '4.0', '5.0', '8.0', '10.0', '9.0', '6.0']
Normal_Nucleoli (numeric, 10 distinct): ['1.0', '10.0', '3.0', '2.0', '8.0', '6.0', '5.0', '4.0', '7.0', '9.0']
Mitoses (numeric, 9 distinct): ['1.0', '2.0', '3.0', '10.0', '4.0', '7.0', '8.0', '5.0', '6.0']
'''

CONTEXT = "Breast Cancer Wisconsin"
TARGET = CuratedTarget(raw_name="Class", new_name="Cancer Type", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
