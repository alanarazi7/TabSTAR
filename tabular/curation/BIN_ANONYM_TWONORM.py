from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: twonorm
====
Examples: 7400
====
URL: https://www.openml.org/search?type=data&id=1507
====
Description: **Author**: Michael Revow     
**Source**: http://www.cs.toronto.edu/~delve/data/twonorm/desc.html  
**Please cite**:     

* Twonorm dataset

This is an implementation of Leo Breiman's twonorm example[1]. It is a 20 dimensional, 2 class classification example. Each class is drawn from a multivariate normal distribution with unit variance. Class 1 has mean (a,a,..a) while Class 2 has mean (-a,-a,..-a). Where a = 2/sqrt(20). Breiman reports the theoretical expected misclassification rate as 2.3%. He used 300 training examples with CART and found an error of 22.1%.
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 6748 distinct): ['-0.052', '0.5468', '-0.7009', '0.09', '0.0439', '-1.5006', '-0.0801', '-0.322', '-0.3698', '0.0383']
V2 (numeric, 6734 distinct): ['1.1282', '-0.764', '-0.0155', '-0.8551', '0.2431', '-0.1574', '0.4399', '-0.0806', '-0.4952', '0.1855']
V3 (numeric, 6768 distinct): ['-0.2553', '-0.663', '0.3899', '-0.2574', '1.1115', '-0.9434', '-0.4553', '0.9883', '0.5754', '1.0678']
V4 (numeric, 6735 distinct): ['-0.6711', '-0.7845', '0.8246', '0.0346', '0.8353', '-0.2728', '-0.3738', '0.3131', '-0.3055', '-0.8697']
V5 (numeric, 6787 distinct): ['-0.0309', '0.9272', '0.2964', '-0.2613', '0.0034', '-0.2437', '-0.4802', '0.0318', '-0.2433', '-0.2061']
V6 (numeric, 6743 distinct): ['-0.5791', '0.3115', '0.6758', '-0.3472', '-0.1793', '-0.2047', '0.663', '1.2778', '1.0274', '-0.3407']
V7 (numeric, 6771 distinct): ['-0.1852', '-0.4113', '0.6469', '0.2907', '0.404', '0.4528', '-0.4664', '-1.0567', '-0.1285', '-0.6819']
V8 (numeric, 6776 distinct): ['0.3209', '-0.2403', '0.204', '-0.1812', '-0.7786', '0.5247', '0.6424', '0.2273', '-0.003', '-0.056']
V9 (numeric, 6742 distinct): ['-0.045', '-0.2271', '0.3169', '-0.385', '0.3468', '0.6493', '-0.2294', '0.0392', '1.7909', '1.186']
V10 (numeric, 6790 distinct): ['-0.3005', '-0.0046', '-0.0434', '0.1148', '-0.271', '1.4894', '0.5436', '-0.471', '0.149', '0.5337']
V11 (numeric, 6764 distinct): ['0.2967', '-0.1941', '-1.2455', '0.2003', '1.5865', '0.0737', '-0.7356', '0.6459', '-0.7518', '-0.5867']
V12 (numeric, 6789 distinct): ['-0.0895', '1.1334', '-0.5541', '-0.4315', '-0.0239', '-0.2499', '-0.6721', '-0.2848', '-0.311', '0.3811']
V13 (numeric, 6731 distinct): ['0.0257', '-0.0203', '0.1128', '-0.3885', '0.5016', '-0.931', '1.3', '-1.0649', '-0.3234', '-0.0651']
V14 (numeric, 6760 distinct): ['-0.1557', '-0.0873', '-0.9664', '-0.4763', '0.7776', '-0.5935', '0.0794', '-0.1261', '-0.3558', '-0.6265']
V15 (numeric, 6732 distinct): ['-0.0443', '0.3082', '-0.2683', '0.0018', '0.772', '0.1128', '-0.4314', '0.4429', '0.4683', '0.795']
V16 (numeric, 6740 distinct): ['0.2635', '0.1596', '-0.3651', '0.2705', '-0.2636', '-0.3889', '-1.125', '-0.2307', '-1.1638', '-0.625']
V17 (numeric, 6775 distinct): ['0.5843', '-0.7982', '0.0643', '-0.5004', '0.9461', '-0.2651', '-1.5947', '0.3306', '-0.6093', '1.0618']
V18 (numeric, 6732 distinct): ['0.116', '-1.2758', '-0.251', '0.0696', '-0.2143', '0.7324', '0.2165', '0.737', '0.6241', '-0.3742']
V19 (numeric, 6706 distinct): ['0.5549', '-1.2586', '-0.2742', '0.1119', '-0.9033', '-0.1763', '0.0324', '0.708', '-1.1476', '-1.7323']
V20 (numeric, 6762 distinct): ['0.3746', '1.5969', '-0.4771', '0.8717', '-0.7708', '-0.5419', '-0.5038', '-0.4027', '0.5661', '-1.6318']
'''

CONTEXT = "Twonorm experiment: multivariate normal distribution with unit variance"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []