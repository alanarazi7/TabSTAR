from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Diabetes(scikit-learn)
====
Examples: 442
====
URL: https://www.openml.org/search?type=data&id=44223
====
Description: .. _diabetes_dataset:

Diabetes dataset
----------------

Ten baseline variables, age, sex, body mass index, average blood
pressure, and six blood serum measurements were obtained for each of n =
442 diabetes patients, as well as the response of interest, a
quantitative measure of disease progression one year after baseline.

**Data Set Characteristics:**

  :Number of Instances: 442

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline

  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, total serum cholesterol
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, total cholesterol / HDL
      - s5      ltg, possibly log of serum triglycerides level
      - s6      glu, blood sugar level

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square root of `n_samples` (i.e. the sum of squares of each column totals 1).

Source URL:
https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

For more information see:
Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
(https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
====
Target Variable: class (numeric, 214 distinct): ['200', '72', '90', '178', '71', '128', '52', '55', '53', '65']
====
Features:

age (numeric, 58 distinct): ['0.0163', '0.0417', '0.009', '-0.0273', '-0.0019', '-0.0527', '0.0453', '0.0126', '0.0671', '0.0054']
sex (numeric, 2 distinct): ['-0.0446', '0.0507']
bmi (numeric, 163 distinct): ['-0.0245', '-0.031', '-0.0084', '-0.0461', '-0.0256', '0.0013', '0.0046', '0.0143', '-0.0202', '-0.0235']
bp (numeric, 100 distinct): ['-0.0401', '-0.0057', '-0.0263', '0.0219', '-0.0332', '-0.0229', '-0.016', '0.0081', '-0.0126', '0.0494']
s1 (numeric, 141 distinct): ['-0.0071', '-0.0373', '0.0122', '0.0204', '0.0012', '0.0246', '-0.025', '-0.0043', '-0.0029', '-0.0098']
s2 (numeric, 302 distinct): ['-0.001', '0.0162', '0.0566', '-0.0248', '-0.047', '-0.0138', '-0.0545', '-0.0217', '0.0046', '0.0375']
s3 (numeric, 63 distinct): ['-0.0139', '-0.0434', '-0.0397', '-0.0029', '-0.0324', '-0.0213', '0.0081', '-0.0287', '-0.0066', '0.0155']
s4 (numeric, 66 distinct): ['-0.0395', '-0.0026', '0.0343', '0.0712', '-0.0764', '0.1081', '0.145', '-0.0376', '0.0159', '-0.0214']
s5 (numeric, 184 distinct): ['-0.0181', '-0.0307', '-0.0412', '-0.0514', '-0.026', '-0.0332', '-0.0109', '-0.0006', '-0.0612', '-0.0236']
s6 (numeric, 56 distinct): ['0.0031', '0.0196', '0.0072', '-0.0011', '-0.0135', '-0.0176', '-0.0384', '-0.0549', '-0.0052', '0.0155']
'''

CONTEXT = "Diabetes Disease Progression Prediction"
TARGET = CuratedTarget(raw_name="class", new_name="Disease Progression after year", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="s1", new_name="Total Serum Cholesterol"),
            CuratedFeature(raw_name="s2", new_name="Low-Density Lipoproteins"),
            CuratedFeature(raw_name="s3", new_name="High-Density Lipoproteins"),
            CuratedFeature(raw_name="s4", new_name="Total Cholesterol / HDL"),
            CuratedFeature(raw_name="s5", new_name="Serum Triglycerides Level"),
            CuratedFeature(raw_name="s6", new_name="Blood Sugar Level")]