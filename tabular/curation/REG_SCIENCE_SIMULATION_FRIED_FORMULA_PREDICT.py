from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: fried
====
Examples: 40768
====
URL: https://www.openml.org/search?type=data&id=564
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

This is an artificial data set used in Friedman (1991) and also
described in Breiman (1996,p.139). The cases are generated using the
following method: Generate the values of 10 attributes, X1, ..., X10
independently each of which uniformly distributed over [0,1]. Obtain
the value of the target variable Y using the equation:

Y = 10 * sin(pi * X1 * X2) + 20 * (X3 - 0.5)^2 + 10 * X4 + 5 * X5 + sigma(0,1)

Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at
http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html
Original source: Breiman (1996, p.139).
Characteristics: 40768 cases, 11 continuous attributes

References

BREIMAN, L. (1996): Bagging Predictors. Machine Learning, 24(3), 123--140. Kluwer Academic Publishers.
FRIEDMAN, J. (1991): Multivariate Adaptative Regression Splines. Annals of Statistics, 19:1, 1--141.
====
Target Variable: Y (numeric, 17396 distinct): ['15.531', '17.71', '15.178', '14.518', '14.658', '17.008', '13.723', '16.13', '12.931', '10.404']
====
Features:

X1 (numeric, 1001 distinct): ['0.026', '0.378', '0.019', '0.867', '0.144', '0.68', '0.437', '0.913', '0.732', '0.361']
X2 (numeric, 1001 distinct): ['0.388', '0.701', '0.926', '0.485', '0.488', '0.183', '0.045', '0.802', '0.367', '0.334']
X3 (numeric, 1001 distinct): ['0.031', '0.979', '0.113', '0.788', '0.853', '0.401', '0.713', '0.502', '0.387', '0.877']
X4 (numeric, 1001 distinct): ['0.75', '0.914', '0.461', '0.486', '0.177', '0.303', '0.841', '0.658', '0.47', '0.068']
X5 (numeric, 1001 distinct): ['0.512', '0.283', '0.276', '0.641', '0.792', '0.899', '0.014', '0.306', '0.925', '0.898']
X6 (numeric, 1001 distinct): ['0.153', '0.496', '0.676', '0.003', '0.199', '0.201', '0.114', '0.571', '0.677', '0.385']
X7 (numeric, 1001 distinct): ['0.294', '0.838', '0.035', '0.124', '0.586', '0.113', '0.594', '0.548', '0.395', '0.514']
X8 (numeric, 1001 distinct): ['0.276', '0.458', '0.209', '0.817', '0.531', '0.077', '0.158', '0.574', '0.816', '0.783']
X9 (numeric, 1001 distinct): ['0.009', '0.169', '0.663', '0.774', '0.949', '0.109', '0.543', '0.219', '0.868', '0.863']
X10 (numeric, 1001 distinct): ['0.024', '0.343', '0.09', '0.418', '0.411', '0.985', '0.541', '0.21', '0.059', '0.177']
'''

CONTEXT = "Synthetic Formula Solving"
TARGET = CuratedTarget(raw_name="Y", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []