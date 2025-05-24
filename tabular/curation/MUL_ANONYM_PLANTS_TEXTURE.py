from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: one-hundred-plants-texture
====
Examples: 1599
====
URL: https://www.openml.org/search?type=data&id=1493
====
Description: **Author**: James Cope, Thibaut Beghin, Paolo Remagnino, Sarah Barman.  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set) - 2010   
**Please cite**: Charles Mallah, James Cope, James Orwell. Plant Leaf Classification Using Probabilistic Integration of Shape, Texture and Margin Features. Signal Processing, Pattern Recognition and Applications, in press. 2013.     

### Description

One-hundred plant species leaves dataset (Class = Texture).
 
### Sources
```
   (a) Original owners of colour Leaves Samples:

 James Cope, Thibaut Beghin, Paolo Remagnino, Sarah Barman.  
 The colour images are not included.  
 The Leaves were collected in the Royal Botanic Gardens, Kew, UK.  
 email: james.cope@kingston.ac.uk  
   
   (b) This dataset consists of work carried out by James Cope, Charles Mallah, and James Orwell.  
 Donor of database Charles Mallah: charles.mallah@kingston.ac.uk; James Cope:  james.cope@kingston.ac.uk  
```

### Dataset Information

The original data directory contains the binary images (masks) of the leaf samples (colour images not included).
There are three features for each image: Shape, Margin and Texture.
For each feature, a 64 element vector is given per leaf sample.
These vectors are taken as a contiguous descriptor (for shape) or histograms (for texture and margin).
So, there are three different files, one for each feature problem:  
 * 'data_Sha_64.txt' -> prediction based on shape
 * 'data_Tex_64.txt' -> prediction based on texture [**dataset provided here**] 
 * 'data_Mar_64.txt' -> prediction based on margin 

Each row has a 64-element feature vector followed by the Class label.
There is a total of 1600 samples with 16 samples per leaf class (100 classes), and no missing values.

### Attributes Information

Three 64 element feature vectors per sample.

### Relevant Papers

Charles Mallah, James Cope, James Orwell. 
Plant Leaf Classification Using Probabilistic Integration of Shape, Texture and Margin Features. 
Signal Processing, Pattern Recognition and Applications, in press.

J. Cope, P. Remagnino, S. Barman, and P. Wilkin.
Plant texture classification using gabor co-occurrences.
Advances in Visual Computing,
pages 699-677, 2010.

T. Beghin, J. Cope, P. Remagnino, and S. Barman.
Shape and texture based plant leaf classification. 
In: Advanced Concepts for Intelligent Vision Systems,
pages 345-353. Springer, 2010.
====
Target Variable: Class (nominal, 100 distinct): ['51', '64', '74', '73', '72', '71', '70', '69', '68', '67']
====
Features:

V1 (numeric, 151 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0039', '0.0049', '0.0068', '0.0078', '0.0059', '0.0098']
V2 (numeric, 91 distinct): ['0.0', '0.001', '0.002', '0.0049', '0.0029', '0.0039', '0.0059', '0.0068', '0.0078', '0.0137']
V3 (numeric, 72 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0039', '0.0098', '0.0078', '0.0059', '0.0107']
V4 (numeric, 101 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0078', '0.0039', '0.0098', '0.0059', '0.0088']
V5 (numeric, 154 distinct): ['0.0', '0.001', '0.002', '0.0059', '0.0029', '0.0049', '0.0068', '0.0078', '0.0039', '0.0107']
V6 (numeric, 97 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0059', '0.0049', '0.0068', '0.0117', '0.0107']
V7 (numeric, 101 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0098', '0.0068', '0.0137', '0.0078']
V8 (numeric, 116 distinct): ['0.0', '0.001', '0.0039', '0.0029', '0.002', '0.0059', '0.0088', '0.0049', '0.0098', '0.0078']
V9 (numeric, 110 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0039', '0.0049', '0.0059', '0.0078', '0.0068', '0.0098']
V10 (numeric, 146 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0078', '0.0059', '0.0088', '0.0117']
V11 (numeric, 140 distinct): ['0.0', '0.001', '0.002', '0.0049', '0.0029', '0.0039', '0.0068', '0.0059', '0.0098', '0.0088']
V12 (numeric, 195 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0068', '0.0059', '0.0049', '0.0117', '0.0039', '0.0225']
V13 (numeric, 76 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0059', '0.0049', '0.0068', '0.0107', '0.0146']
V14 (numeric, 91 distinct): ['0.0', '0.001', '0.0059', '0.0039', '0.0068', '0.0107', '0.002', '0.0029', '0.0078', '0.0098']
V15 (numeric, 104 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0059', '0.0098', '0.0068', '0.0156']
V16 (numeric, 63 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0068', '0.0029', '0.0059', '0.0049', '0.0146', '0.0088']
V17 (numeric, 116 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0068', '0.0049', '0.0059', '0.0078', '0.0137']
V18 (numeric, 73 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0059', '0.0068', '0.0078', '0.0088']
V19 (numeric, 134 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0039', '0.0068', '0.0049', '0.0059', '0.0078', '0.0117']
V20 (numeric, 92 distinct): ['0.0', '0.0039', '0.0059', '0.0029', '0.0088', '0.001', '0.0078', '0.002', '0.0049', '0.0068']
V21 (numeric, 64 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0088', '0.0078', '0.0127', '0.0098', '0.0039', '0.0176']
V22 (numeric, 101 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0156', '0.0098', '0.0059', '0.0186']
V23 (numeric, 114 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0059', '0.0049', '0.0068', '0.0078', '0.0088']
V24 (numeric, 95 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0059', '0.0039', '0.0068', '0.0088', '0.0078']
V25 (numeric, 82 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0059', '0.0049', '0.0088', '0.0068', '0.0078']
V26 (numeric, 164 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0059', '0.0049', '0.0078', '0.0068', '0.0107']
V27 (numeric, 138 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0068', '0.0059', '0.0107', '0.0088']
V28 (numeric, 94 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0059', '0.0039', '0.0078', '0.0049', '0.0088', '0.0068']
V29 (numeric, 119 distinct): ['0.0', '0.0029', '0.0039', '0.001', '0.0078', '0.002', '0.0098', '0.0088', '0.0068', '0.0059']
V30 (numeric, 67 distinct): ['0.0', '0.001', '0.002', '0.0059', '0.0049', '0.0039', '0.0029', '0.0088', '0.0068', '0.0117']
V31 (numeric, 124 distinct): ['0.0', '0.002', '0.001', '0.0039', '0.0049', '0.0029', '0.0059', '0.0107', '0.0088', '0.0186']
V32 (numeric, 84 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0098', '0.0127', '0.0078', '0.0059']
V33 (numeric, 184 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0059', '0.0068', '0.0049', '0.0098', '0.0137']
V34 (numeric, 170 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0068', '0.0049', '0.0088', '0.0117', '0.0166', '0.0059']
V35 (numeric, 65 distinct): ['0.0', '0.002', '0.001', '0.0068', '0.0039', '0.0137', '0.0059', '0.0088', '0.0049', '0.0098']
V36 (numeric, 73 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0059', '0.0215', '0.0068', '0.0049', '0.0137']
V37 (numeric, 173 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0068', '0.0078', '0.0088', '0.0059']
V38 (numeric, 121 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0205', '0.0195', '0.0283', '0.0098', '0.0176']
V39 (numeric, 100 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0039', '0.0059', '0.0088', '0.0166', '0.0068']
V40 (numeric, 118 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0078', '0.0068', '0.0088', '0.0039', '0.0098']
V41 (numeric, 152 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0059', '0.0068', '0.0088', '0.0107']
V42 (numeric, 61 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0049', '0.0039', '0.0068', '0.0059', '0.0088', '0.0078']
V43 (numeric, 88 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0068', '0.0049', '0.0078', '0.0088', '0.0127']
V44 (numeric, 176 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0059', '0.0039', '0.0049', '0.0068', '0.0078', '0.0107']
V45 (numeric, 101 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0068', '0.0088', '0.0059', '0.0049', '0.0117']
V46 (numeric, 121 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0107', '0.0303', '0.0059', '0.0078', '0.0088']
V47 (numeric, 77 distinct): ['0.0', '0.0127', '0.0117', '0.0049', '0.0039', '0.001', '0.0078', '0.002', '0.0205', '0.0098']
V48 (numeric, 131 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0107', '0.0254', '0.0049', '0.0059', '0.0078']
V49 (numeric, 73 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0049', '0.0039', '0.0068', '0.0059', '0.0098', '0.0078']
V50 (numeric, 129 distinct): ['0.0', '0.0029', '0.001', '0.002', '0.0068', '0.0088', '0.0059', '0.0049', '0.0078', '0.0137']
V51 (numeric, 132 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0059', '0.0098', '0.0078', '0.0088']
V52 (numeric, 73 distinct): ['0.0', '0.001', '0.002', '0.0088', '0.0029', '0.0146', '0.0078', '0.0059', '0.0098', '0.0049']
V53 (numeric, 98 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0068', '0.0039', '0.0049', '0.0078', '0.0059', '0.0088']
V54 (numeric, 119 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0049', '0.0068', '0.0059', '0.0078', '0.0098']
V55 (numeric, 224 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0049', '0.0098', '0.0078', '0.0068', '0.0088']
V56 (numeric, 91 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0107', '0.0078', '0.0059', '0.0068']
V57 (numeric, 109 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0039', '0.0068', '0.0059', '0.0088', '0.0078']
V58 (numeric, 116 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0039', '0.0059', '0.0068', '0.0088', '0.0117']
V59 (numeric, 85 distinct): ['0.0', '0.0029', '0.0078', '0.0059', '0.001', '0.0117', '0.002', '0.0137', '0.0186', '0.0127']
V60 (numeric, 125 distinct): ['0.0', '0.001', '0.002', '0.0039', '0.0029', '0.0068', '0.0117', '0.0049', '0.0059', '0.0156']
V61 (numeric, 64 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0049', '0.0469', '0.0352', '0.0127', '0.0371', '0.0078']
V62 (numeric, 141 distinct): ['0.0', '0.001', '0.002', '0.0029', '0.0039', '0.0049', '0.0059', '0.0068', '0.0078', '0.0098']
V63 (numeric, 70 distinct): ['0.0', '0.0039', '0.001', '0.0029', '0.002', '0.0059', '0.0049', '0.0088', '0.0068', '0.0098']
V64 (numeric, 108 distinct): ['0.0', '0.001', '0.0029', '0.002', '0.0049', '0.0059', '0.0088', '0.0117', '0.0225', '0.0127']
'''

CONTEXT = "Anonymized: Image data of plant leaves with texture features"
TARGET = CuratedTarget(raw_name="Class", new_name="Texture", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []