from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: optdigits
====
Examples: 5620
====
URL: https://www.openml.org/search?type=data&id=28
====
Description: **Author**: E. Alpaydin, C. Kaynak  
**Source**: [UCI](http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits)  
**Please cite**: [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)  

1. Title of Database: Optical Recognition of Handwritten Digits
 
 2. Source:
  E. Alpaydin, C. Kaynak
  Department of Computer Engineering
  Bogazici University, 80815 Istanbul Turkey
  alpaydin@boun.edu.tr
  July 1998
 
 3. Past Usage:
  C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
  Applications to Handwritten Digit Recognition, 
  MSc Thesis, Institute of Graduate Studies in Science and 
  Engineering, Bogazici University.
 
  E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika,
  to appear. ftp://ftp.icsi.berkeley.edu/pub/ai/ethem/kyb.ps.Z
 
 4. Relevant Information:
  We used preprocessing programs made available by NIST to extract
  normalized bitmaps of handwritten digits from a preprinted form. From
  a total of 43 people, 30 contributed to the training set and different
  13 to the test set. 32x32 bitmaps are divided into nonoverlapping 
  blocks of 4x4 and the number of on pixels are counted in each block.
  This generates an input matrix of 8x8 where each element is an 
  integer in the range 0..16. This reduces dimensionality and gives 
  invariance to small distortions.
 
  For info on NIST preprocessing routines, see 
  M. D. Garris, J. L. Blue, G. T. Candela, D. L. Dimmick, J. Geist, 
  P. J. Grother, S. A. Janet, and C. L. Wilson, NIST Form-Based 
  Handprint Recognition System, NISTIR 5469, 1994.
 
 5. Number of Instances
  optdigits.tra Training 3823
  optdigits.tes Testing  1797
  
  The way we used the dataset was to use half of training for 
  actual training, one-fourth for validation and one-fourth
  for writer-dependent testing. The test set was used for 
  writer-independent testing and is the actual quality measure.
 
 6. Number of Attributes
  64 input+1 class attribute
 
 7. For Each Attribute:
  All input attributes are integers in the range 0..16.
  The last attribute is the class code 0..9
 
 8. Missing Attribute Values
  None
 
 9. Class Distribution
  Class: No of examples in training set
  0:  376
  1:  389
  2:  380
  3:  389
  4:  387
  5:  376
  6:  377
  7:  387
  8:  380
  9:  382
 
  Class: No of examples in testing set
  0:  178
  1:  182
  2:  177
  3:  183
  4:  181
  5:  182
  6:  181
  7:  179
  8:  174
  9:  180
 
 Accuracy on the testing set with k-nn 
 using Euclidean distance as the metric
 
  k =  1   : 98.00
  k =  2   : 97.38
  k =  3   : 97.83
  k =  4   : 97.61
  k =  5   : 97.89
  k =  6   : 97.77
  k =  7   : 97.66
  k =  8   : 97.66
  k =  9   : 97.72
  k = 10   : 97.55
  k = 11   : 97.89
====
Target Variable: class (nominal, 10 distinct): ['3', '1', '4', '7', '9', '5', '6', '2', '0', '8']
====
Features:

input1 (numeric, 1 distinct): ['0.0']
input2 (numeric, 9 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
input3 (numeric, 17 distinct): ['0.0', '2.0', '6.0', '5.0', '3.0', '1.0', '4.0', '7.0', '8.0', '9.0']
input4 (numeric, 17 distinct): ['16.0', '15.0', '12.0', '14.0', '13.0', '11.0', '10.0', '9.0', '8.0', '0.0']
input5 (numeric, 17 distinct): ['16.0', '15.0', '13.0', '12.0', '14.0', '11.0', '10.0', '8.0', '9.0', '0.0']
input6 (numeric, 17 distinct): ['0.0', '1.0', '16.0', '2.0', '3.0', '4.0', '5.0', '12.0', '6.0', '8.0']
input7 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '7.0', '6.0', '16.0', '8.0']
input8 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0', '4.0', '8.0', '6.0', '12.0', '10.0']
input9 (numeric, 4 distinct): ['0.0', '1.0', '2.0', '5.0']
input10 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '7.0', '6.0', '8.0', '10.0']
input11 (numeric, 17 distinct): ['16.0', '15.0', '0.0', '14.0', '13.0', '12.0', '11.0', '10.0', '8.0', '9.0']
input12 (numeric, 17 distinct): ['16.0', '15.0', '14.0', '13.0', '12.0', '8.0', '9.0', '10.0', '11.0', '7.0']
input13 (numeric, 17 distinct): ['16.0', '8.0', '15.0', '12.0', '14.0', '13.0', '11.0', '10.0', '9.0', '4.0']
input14 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '12.0', '13.0', '14.0', '8.0', '11.0', '4.0', '10.0']
input15 (numeric, 17 distinct): ['0.0', '2.0', '1.0', '3.0', '4.0', '5.0', '8.0', '7.0', '16.0', '6.0']
input16 (numeric, 15 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0', '7.0', '6.0', '4.0', '8.0', '9.0']
input17 (numeric, 5 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0']
input18 (numeric, 17 distinct): ['0.0', '2.0', '1.0', '4.0', '3.0', '5.0', '8.0', '6.0', '7.0', '9.0']
input19 (numeric, 17 distinct): ['16.0', '0.0', '15.0', '14.0', '12.0', '13.0', '11.0', '10.0', '8.0', '9.0']
input20 (numeric, 17 distinct): ['0.0', '16.0', '1.0', '2.0', '4.0', '3.0', '5.0', '8.0', '15.0', '6.0']
input21 (numeric, 17 distinct): ['0.0', '16.0', '1.0', '2.0', '14.0', '4.0', '12.0', '3.0', '5.0', '13.0']
input22 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '14.0', '12.0', '13.0', '8.0', '11.0', '10.0', '9.0']
input23 (numeric, 17 distinct): ['0.0', '4.0', '2.0', '1.0', '3.0', '6.0', '5.0', '8.0', '7.0', '10.0']
input24 (numeric, 9 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '7.0']
input25 (numeric, 2 distinct): ['0.0', '1.0']
input26 (numeric, 17 distinct): ['0.0', '1.0', '4.0', '2.0', '5.0', '3.0', '6.0', '8.0', '7.0', '9.0']
input27 (numeric, 17 distinct): ['16.0', '0.0', '15.0', '14.0', '12.0', '13.0', '11.0', '8.0', '1.0', '2.0']
input28 (numeric, 17 distinct): ['16.0', '0.0', '12.0', '15.0', '14.0', '8.0', '13.0', '4.0', '10.0', '11.0']
input29 (numeric, 17 distinct): ['16.0', '0.0', '15.0', '14.0', '12.0', '13.0', '8.0', '11.0', '10.0', '1.0']
input30 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '12.0', '14.0', '13.0', '8.0', '1.0', '3.0', '6.0']
input31 (numeric, 17 distinct): ['0.0', '8.0', '1.0', '2.0', '6.0', '4.0', '5.0', '3.0', '7.0', '9.0']
input32 (numeric, 3 distinct): ['0.0', '1.0', '2.0']
input33 (numeric, 2 distinct): ['0.0', '1.0']
input34 (numeric, 16 distinct): ['0.0', '1.0', '8.0', '3.0', '5.0', '2.0', '4.0', '7.0', '6.0', '9.0']
input35 (numeric, 17 distinct): ['0.0', '16.0', '12.0', '15.0', '8.0', '1.0', '13.0', '14.0', '2.0', '7.0']
input36 (numeric, 17 distinct): ['16.0', '0.0', '15.0', '14.0', '8.0', '12.0', '13.0', '4.0', '11.0', '10.0']
input37 (numeric, 17 distinct): ['16.0', '0.0', '15.0', '14.0', '12.0', '13.0', '8.0', '10.0', '11.0', '9.0']
input38 (numeric, 17 distinct): ['16.0', '0.0', '15.0', '14.0', '12.0', '13.0', '8.0', '10.0', '11.0', '9.0']
input39 (numeric, 15 distinct): ['0.0', '8.0', '1.0', '2.0', '4.0', '5.0', '3.0', '6.0', '7.0', '9.0']
input40 (numeric, 1 distinct): ['0.0']
input41 (numeric, 8 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0']
input42 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '4.0', '3.0', '5.0', '6.0', '8.0', '7.0', '9.0']
input43 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '14.0', '13.0', '12.0', '1.0', '4.0', '2.0', '3.0']
input44 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '14.0', '1.0', '13.0', '12.0', '8.0', '9.0', '4.0']
input45 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '1.0', '12.0', '13.0', '14.0', '8.0', '4.0', '2.0']
input46 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '14.0', '13.0', '12.0', '11.0', '8.0', '9.0', '10.0']
input47 (numeric, 17 distinct): ['0.0', '1.0', '8.0', '3.0', '7.0', '5.0', '4.0', '2.0', '6.0', '9.0']
input48 (numeric, 7 distinct): ['0.0', '1.0', '2.0', '4.0', '6.0', '3.0', '5.0']
input49 (numeric, 9 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0', '10.0', '7.0', '4.0', '8.0']
input50 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '7.0', '6.0', '8.0', '9.0']
input51 (numeric, 17 distinct): ['0.0', '16.0', '11.0', '12.0', '14.0', '15.0', '8.0', '13.0', '9.0', '10.0']
input52 (numeric, 17 distinct): ['16.0', '8.0', '0.0', '14.0', '12.0', '4.0', '13.0', '15.0', '10.0', '11.0']
input53 (numeric, 17 distinct): ['16.0', '15.0', '0.0', '8.0', '14.0', '4.0', '13.0', '12.0', '5.0', '7.0']
input54 (numeric, 17 distinct): ['0.0', '16.0', '15.0', '14.0', '12.0', '13.0', '11.0', '8.0', '10.0', '9.0']
input55 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '4.0', '3.0', '7.0', '8.0', '5.0', '6.0', '16.0']
input56 (numeric, 13 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0', '4.0', '6.0', '8.0', '7.0', '10.0']
input57 (numeric, 2 distinct): ['0.0', '1.0']
input58 (numeric, 11 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '9.0', '8.0']
input59 (numeric, 17 distinct): ['0.0', '2.0', '1.0', '5.0', '3.0', '4.0', '6.0', '7.0', '8.0', '10.0']
input60 (numeric, 17 distinct): ['16.0', '15.0', '13.0', '14.0', '12.0', '11.0', '10.0', '0.0', '9.0', '8.0']
input61 (numeric, 17 distinct): ['16.0', '15.0', '12.0', '14.0', '13.0', '0.0', '11.0', '10.0', '9.0', '8.0']
input62 (numeric, 17 distinct): ['0.0', '16.0', '12.0', '1.0', '8.0', '15.0', '3.0', '9.0', '10.0', '13.0']
input63 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '16.0', '5.0', '7.0', '6.0', '8.0']
input64 (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0', '6.0', '4.0', '7.0', '8.0', '11.0']
'''

CONTEXT = "Opt Digits Image Recognition"
TARGET = CuratedTarget(raw_name="class", new_name="Digit", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []