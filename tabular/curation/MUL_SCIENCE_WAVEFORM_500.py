from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: waveform-5000
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=60
====
Description: **Author**: Breiman,L., Friedman,J.H., Olshen,R.A., & Stone,C.J.  
**Source**: [UCI](http://archive.ics.uci.edu/ml/datasets/waveform+database+generator+(version+2)) - 1988  
**Please cite**: [UCI](http://archive.ics.uci.edu/ml/citation_policy.html)    

**Waveform Database Generator**  
Generator generating 3 classes of waves. Each class is generated from a combination of 2 of 3 "base" waves.  

For details, see Breiman,L., Friedman,J.H., Olshen,R.A., and Stone,C.J. (1984). 
Classification and Regression Trees. Wadsworth International, pp 49-55, 169. 

Note: There is [an earlier version](http://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+(Version+1)) of this dataset that only has 21 attributes (it does not add the 19 noise features).

### Attribute Information

40 attributes describing the waveform, all of which include noise. The latter 19 attributes are all noise attributes with mean 0 and variance 1.
====
Target Variable: class (nominal, 3 distinct): ['0', '2', '1']
====
Features:

x1 (numeric, 529 distinct): ['-0.23', '0.48', '-0.63', '-0.12', '-0.3', '-0.11', '-0.29', '0.12', '0.31', '0.23']
x2 (numeric, 541 distinct): ['0.68', '0.35', '0.03', '0.79', '0.78', '0.22', '0.26', '0.18', '0.42', '0.8']
x3 (numeric, 614 distinct): ['0.82', '0.34', '0.87', '0.6', '0.86', '1.49', '0.38', '0.64', '0.12', '0.39']
x4 (numeric, 691 distinct): ['0.5', '0.83', '1.16', '0.01', '0.28', '1.73', '-0.15', '0.35', '0.59', '0.43']
x5 (numeric, 772 distinct): ['1.47', '0.4', '-0.23', '0.63', '1.9', '-0.03', '1.68', '-0.45', '0.67', '0.85']
x6 (numeric, 815 distinct): ['1.48', '1.68', '2.65', '1.33', '1.0', '1.67', '0.31', '1.02', '2.41', '0.83']
x7 (numeric, 890 distinct): ['1.67', '4.04', '1.57', '1.58', '2.02', '1.37', '2.08', '1.62', '3.51', '2.32']
x8 (numeric, 802 distinct): ['2.72', '3.89', '1.9', '4.16', '3.58', '2.03', '3.51', '3.4', '3.63', '3.71']
x9 (numeric, 764 distinct): ['2.11', '3.88', '3.27', '2.17', '3.51', '3.11', '4.13', '3.41', '3.13', '3.37']
x10 (numeric, 713 distinct): ['2.79', '3.47', '2.9', '3.54', '3.23', '4.3', '3.35', '3.39', '2.4', '1.74']
x11 (numeric, 749 distinct): ['2.11', '2.33', '2.52', '2.51', '3.78', '3.19', '3.26', '2.46', '2.47', '2.98']
x12 (numeric, 730 distinct): ['2.68', '3.03', '3.31', '3.07', '3.05', '3.41', '2.6', '2.86', '2.48', '2.55']
x13 (numeric, 768 distinct): ['4.04', '3.13', '2.1', '3.8', '3.37', '3.1', '2.77', '2.34', '1.42', '2.07']
x14 (numeric, 793 distinct): ['3.02', '2.87', '1.89', '3.56', '3.79', '3.84', '3.58', '2.91', '1.38', '3.75']
x15 (numeric, 887 distinct): ['1.24', '1.51', '1.28', '1.55', '1.07', '1.72', '1.82', '0.94', '2.76', '1.56']
x16 (numeric, 816 distinct): ['0.83', '1.36', '0.06', '1.89', '1.47', '1.18', '1.09', '0.39', '1.17', '1.4']
x17 (numeric, 754 distinct): ['0.52', '0.4', '-0.28', '0.41', '1.56', '0.29', '0.08', '0.55', '1.05', '0.67']
x18 (numeric, 681 distinct): ['0.61', '0.86', '0.58', '0.4', '1.92', '1.14', '1.07', '0.24', '1.29', '0.46']
x19 (numeric, 605 distinct): ['0.73', '0.88', '0.06', '0.25', '-0.01', '0.78', '0.59', '0.08', '0.14', '0.45']
x20 (numeric, 554 distinct): ['-0.08', '0.44', '0.36', '0.84', '-0.1', '0.74', '0.64', '0.07', '0.39', '0.25']
x21 (numeric, 531 distinct): ['0.37', '-0.33', '-0.41', '-0.11', '0.07', '0.3', '0.06', '0.43', '0.5', '0.44']
x22 (numeric, 527 distinct): ['0.07', '-0.21', '0.25', '0.11', '0.14', '0.32', '0.22', '-0.04', '-0.26', '0.27']
x23 (numeric, 516 distinct): ['-0.16', '-0.41', '-0.14', '0.1', '-0.37', '-0.17', '-0.29', '-0.03', '0.58', '-0.15']
x24 (numeric, 524 distinct): ['-0.44', '-0.43', '0.04', '0.39', '0.03', '-0.26', '-0.42', '0.35', '-0.2', '0.4']
x25 (numeric, 527 distinct): ['0.06', '-0.14', '-0.38', '0.26', '0.08', '0.32', '-0.24', '0.24', '-0.13', '0.85']
x26 (numeric, 523 distinct): ['-0.2', '0.19', '0.49', '-0.15', '-0.19', '0.06', '0.81', '-0.1', '-0.22', '0.26']
x27 (numeric, 525 distinct): ['0.05', '-0.02', '0.04', '0.15', '-0.21', '0.25', '0.35', '-0.34', '-0.22', '-0.16']
x28 (numeric, 534 distinct): ['-0.16', '-0.43', '0.1', '0.39', '-0.34', '0.45', '0.2', '-0.15', '0.04', '0.33']
x29 (numeric, 527 distinct): ['0.33', '0.13', '0.48', '-0.11', '-0.06', '-0.17', '-0.29', '0.27', '0.23', '-0.12']
x30 (numeric, 525 distinct): ['0.59', '-0.11', '-0.02', '0.26', '0.13', '-0.4', '0.03', '-0.28', '0.15', '0.3']
x31 (numeric, 535 distinct): ['-0.32', '0.28', '0.11', '-0.2', '0.31', '0.22', '0.59', '0.13', '-0.6', '0.27']
x32 (numeric, 530 distinct): ['0.01', '-0.11', '0.14', '0.29', '0.04', '0.2', '0.81', '-0.07', '0.4', '0.02']
x33 (numeric, 528 distinct): ['0.58', '-0.03', '0.18', '0.09', '0.08', '-0.52', '0.03', '-0.07', '0.2', '0.43']
x34 (numeric, 522 distinct): ['-0.27', '0.03', '0.24', '0.31', '-0.07', '0.23', '0.01', '0.07', '-0.18', '0.45']
x35 (numeric, 537 distinct): ['-0.14', '-0.02', '0.28', '-0.18', '0.05', '-0.04', '-0.33', '0.0', '0.8', '-0.63']
x36 (numeric, 515 distinct): ['-0.16', '0.26', '0.01', '-0.33', '-0.58', '0.19', '0.04', '0.05', '-0.48', '-0.23']
x37 (numeric, 511 distinct): ['-0.03', '0.28', '-0.01', '-0.27', '-0.17', '0.59', '-0.13', '0.26', '-0.15', '-0.21']
x38 (numeric, 520 distinct): ['0.71', '-0.05', '-0.58', '0.23', '-0.07', '0.12', '-0.44', '0.05', '-0.04', '-0.09']
x39 (numeric, 531 distinct): ['0.14', '0.05', '-0.33', '-0.34', '-0.51', '0.38', '0.13', '0.39', '-0.04', '-0.03']
x40 (numeric, 530 distinct): ['0.1', '-0.19', '0.0', '0.59', '-0.26', '-0.04', '0.4', '0.51', '-0.35', '-0.65']
'''

CONTEXT = "Waveform Database Generator"
TARGET = CuratedTarget(raw_name="class", new_name="Result", task_type=SupervisedTask.MULTICLASS,)
COLS_TO_DROP = []
FEATURES = []