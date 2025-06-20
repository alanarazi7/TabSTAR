from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: dna
====
Examples: 3186
====
URL: https://www.openml.org/search?type=data&id=40670
====
Description: **Author**: Ross King, based on data from Genbank 64.1  
**Source**: [MLbench](https://www.rdocumentation.org/packages/mlbench/versions/2.1-1/topics/DNA). Originally from the StatLog project.  
**Please Cite**:   

**Primate Splice-Junction Gene Sequences (DNA)**  
Originally from the StatLog project. The raw data is still available on [UCI](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)).

The data consists of 3,186 data points (splice junctions). The data points are described by 180 indicator binary variables and the problem is to recognize the 3 classes (ei, ie, neither), i.e., the boundaries between exons (the parts of the DNA sequence retained after splicing) and introns (the parts of the DNA sequence that are spliced out). The StatLog DNA dataset is a processed version of the [Irvine database]((https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences))). The main difference is that the symbolic variables representing the nucleotides (only A,G,T,C) were replaced by 3 binary indicator variables. Thus the original 60 symbolic attributes were changed into 180 binary attributes. The names of the examples were removed. The examples with ambiguities were removed (there was very few of them, 4). The StatLog version of this dataset was produced by Ross King at Strathclyde University. For original details see the Irvine database documentation.

The nucleotides A,C,G,T were given indicator values as follows:
A -> 1 0 0
C -> 0 1 0
G -> 0 0 1
T -> 0 0 0

Hint: Much better performance is generally observed if attributes closest to the junction are used. In the StatLog version, this means using attributes A61 to A120 only.
====
Target Variable: class (nominal, 3 distinct): ['3', '1', '2']
====
Features:

A0 (nominal, 2 distinct): ['0', '1']
A1 (nominal, 2 distinct): ['0', '1']
A2 (nominal, 2 distinct): ['0', '1']
A3 (nominal, 2 distinct): ['0', '1']
A4 (nominal, 2 distinct): ['0', '1']
A5 (nominal, 2 distinct): ['0', '1']
A6 (nominal, 2 distinct): ['0', '1']
A7 (nominal, 2 distinct): ['0', '1']
A8 (nominal, 2 distinct): ['0', '1']
A9 (nominal, 2 distinct): ['0', '1']
A10 (nominal, 2 distinct): ['0', '1']
A11 (nominal, 2 distinct): ['0', '1']
A12 (nominal, 2 distinct): ['0', '1']
A13 (nominal, 2 distinct): ['0', '1']
A14 (nominal, 2 distinct): ['0', '1']
A15 (nominal, 2 distinct): ['0', '1']
A16 (nominal, 2 distinct): ['0', '1']
A17 (nominal, 2 distinct): ['0', '1']
A18 (nominal, 2 distinct): ['0', '1']
A19 (nominal, 2 distinct): ['0', '1']
A20 (nominal, 2 distinct): ['0', '1']
A21 (nominal, 2 distinct): ['0', '1']
A22 (nominal, 2 distinct): ['0', '1']
A23 (nominal, 2 distinct): ['0', '1']
A24 (nominal, 2 distinct): ['0', '1']
A25 (nominal, 2 distinct): ['0', '1']
A26 (nominal, 2 distinct): ['0', '1']
A27 (nominal, 2 distinct): ['0', '1']
A28 (nominal, 2 distinct): ['0', '1']
A29 (nominal, 2 distinct): ['0', '1']
A30 (nominal, 2 distinct): ['0', '1']
A31 (nominal, 2 distinct): ['0', '1']
A32 (nominal, 2 distinct): ['0', '1']
A33 (nominal, 2 distinct): ['0', '1']
A34 (nominal, 2 distinct): ['0', '1']
A35 (nominal, 2 distinct): ['0', '1']
A36 (nominal, 2 distinct): ['0', '1']
A37 (nominal, 2 distinct): ['0', '1']
A38 (nominal, 2 distinct): ['0', '1']
A39 (nominal, 2 distinct): ['0', '1']
A40 (nominal, 2 distinct): ['0', '1']
A41 (nominal, 2 distinct): ['0', '1']
A42 (nominal, 2 distinct): ['0', '1']
A43 (nominal, 2 distinct): ['0', '1']
A44 (nominal, 2 distinct): ['0', '1']
A45 (nominal, 2 distinct): ['0', '1']
A46 (nominal, 2 distinct): ['0', '1']
A47 (nominal, 2 distinct): ['0', '1']
A48 (nominal, 2 distinct): ['0', '1']
A49 (nominal, 2 distinct): ['0', '1']
A50 (nominal, 2 distinct): ['0', '1']
A51 (nominal, 2 distinct): ['0', '1']
A52 (nominal, 2 distinct): ['0', '1']
A53 (nominal, 2 distinct): ['0', '1']
A54 (nominal, 2 distinct): ['0', '1']
A55 (nominal, 2 distinct): ['0', '1']
A56 (nominal, 2 distinct): ['0', '1']
A57 (nominal, 2 distinct): ['0', '1']
A58 (nominal, 2 distinct): ['0', '1']
A59 (nominal, 2 distinct): ['0', '1']
A60 (nominal, 2 distinct): ['0', '1']
A61 (nominal, 2 distinct): ['0', '1']
A62 (nominal, 2 distinct): ['0', '1']
A63 (nominal, 2 distinct): ['0', '1']
A64 (nominal, 2 distinct): ['0', '1']
A65 (nominal, 2 distinct): ['0', '1']
A66 (nominal, 2 distinct): ['0', '1']
A67 (nominal, 2 distinct): ['0', '1']
A68 (nominal, 2 distinct): ['0', '1']
A69 (nominal, 2 distinct): ['0', '1']
A70 (nominal, 2 distinct): ['0', '1']
A71 (nominal, 2 distinct): ['0', '1']
A72 (nominal, 2 distinct): ['0', '1']
A73 (nominal, 2 distinct): ['0', '1']
A74 (nominal, 2 distinct): ['0', '1']
A75 (nominal, 2 distinct): ['0', '1']
A76 (nominal, 2 distinct): ['0', '1']
A77 (nominal, 2 distinct): ['0', '1']
A78 (nominal, 2 distinct): ['0', '1']
A79 (nominal, 2 distinct): ['0', '1']
A80 (nominal, 2 distinct): ['0', '1']
A81 (nominal, 2 distinct): ['0', '1']
A82 (nominal, 2 distinct): ['0', '1']
A83 (nominal, 2 distinct): ['0', '1']
A84 (nominal, 2 distinct): ['1', '0']
A85 (nominal, 2 distinct): ['0', '1']
A86 (nominal, 2 distinct): ['0', '1']
A87 (nominal, 2 distinct): ['0', '1']
A88 (nominal, 2 distinct): ['0', '1']
A89 (nominal, 2 distinct): ['1', '0']
A90 (nominal, 2 distinct): ['0', '1']
A91 (nominal, 2 distinct): ['0', '1']
A92 (nominal, 2 distinct): ['0', '1']
A93 (nominal, 2 distinct): ['0', '1']
A94 (nominal, 2 distinct): ['0', '1']
A95 (nominal, 2 distinct): ['0', '1']
A96 (nominal, 2 distinct): ['0', '1']
A97 (nominal, 2 distinct): ['0', '1']
A98 (nominal, 2 distinct): ['0', '1']
A99 (nominal, 2 distinct): ['0', '1']
A100 (nominal, 2 distinct): ['0', '1']
A101 (nominal, 2 distinct): ['0', '1']
A102 (nominal, 2 distinct): ['0', '1']
A103 (nominal, 2 distinct): ['0', '1']
A104 (nominal, 2 distinct): ['0', '1']
A105 (nominal, 2 distinct): ['0', '1']
A106 (nominal, 2 distinct): ['0', '1']
A107 (nominal, 2 distinct): ['0', '1']
A108 (nominal, 2 distinct): ['0', '1']
A109 (nominal, 2 distinct): ['0', '1']
A110 (nominal, 2 distinct): ['0', '1']
A111 (nominal, 2 distinct): ['0', '1']
A112 (nominal, 2 distinct): ['0', '1']
A113 (nominal, 2 distinct): ['0', '1']
A114 (nominal, 2 distinct): ['0', '1']
A115 (nominal, 2 distinct): ['0', '1']
A116 (nominal, 2 distinct): ['0', '1']
A117 (nominal, 2 distinct): ['0', '1']
A118 (nominal, 2 distinct): ['0', '1']
A119 (nominal, 2 distinct): ['0', '1']
A120 (nominal, 2 distinct): ['0', '1']
A121 (nominal, 2 distinct): ['0', '1']
A122 (nominal, 2 distinct): ['0', '1']
A123 (nominal, 2 distinct): ['0', '1']
A124 (nominal, 2 distinct): ['0', '1']
A125 (nominal, 2 distinct): ['0', '1']
A126 (nominal, 2 distinct): ['0', '1']
A127 (nominal, 2 distinct): ['0', '1']
A128 (nominal, 2 distinct): ['0', '1']
A129 (nominal, 2 distinct): ['0', '1']
A130 (nominal, 2 distinct): ['0', '1']
A131 (nominal, 2 distinct): ['0', '1']
A132 (nominal, 2 distinct): ['0', '1']
A133 (nominal, 2 distinct): ['0', '1']
A134 (nominal, 2 distinct): ['0', '1']
A135 (nominal, 2 distinct): ['0', '1']
A136 (nominal, 2 distinct): ['0', '1']
A137 (nominal, 2 distinct): ['0', '1']
A138 (nominal, 2 distinct): ['0', '1']
A139 (nominal, 2 distinct): ['0', '1']
A140 (nominal, 2 distinct): ['0', '1']
A141 (nominal, 2 distinct): ['0', '1']
A142 (nominal, 2 distinct): ['0', '1']
A143 (nominal, 2 distinct): ['0', '1']
A144 (nominal, 2 distinct): ['0', '1']
A145 (nominal, 2 distinct): ['0', '1']
A146 (nominal, 2 distinct): ['0', '1']
A147 (nominal, 2 distinct): ['0', '1']
A148 (nominal, 2 distinct): ['0', '1']
A149 (nominal, 2 distinct): ['0', '1']
A150 (nominal, 2 distinct): ['0', '1']
A151 (nominal, 2 distinct): ['0', '1']
A152 (nominal, 2 distinct): ['0', '1']
A153 (nominal, 2 distinct): ['0', '1']
A154 (nominal, 2 distinct): ['0', '1']
A155 (nominal, 2 distinct): ['0', '1']
A156 (nominal, 2 distinct): ['0', '1']
A157 (nominal, 2 distinct): ['0', '1']
A158 (nominal, 2 distinct): ['0', '1']
A159 (nominal, 2 distinct): ['0', '1']
A160 (nominal, 2 distinct): ['0', '1']
A161 (nominal, 2 distinct): ['0', '1']
A162 (nominal, 2 distinct): ['0', '1']
A163 (nominal, 2 distinct): ['0', '1']
A164 (nominal, 2 distinct): ['0', '1']
A165 (nominal, 2 distinct): ['0', '1']
A166 (nominal, 2 distinct): ['0', '1']
A167 (nominal, 2 distinct): ['0', '1']
A168 (nominal, 2 distinct): ['0', '1']
A169 (nominal, 2 distinct): ['0', '1']
A170 (nominal, 2 distinct): ['0', '1']
A171 (nominal, 2 distinct): ['0', '1']
A172 (nominal, 2 distinct): ['0', '1']
A173 (nominal, 2 distinct): ['0', '1']
A174 (nominal, 2 distinct): ['0', '1']
A175 (nominal, 2 distinct): ['0', '1']
A176 (nominal, 2 distinct): ['0', '1']
A177 (nominal, 2 distinct): ['0', '1']
A178 (nominal, 2 distinct): ['0', '1']
A179 (nominal, 2 distinct): ['0', '1']
'''

CONTEXT = "DNA Sequences"
TARGET = CuratedTarget(raw_name="class", new_name="Boundaries Between Exons and Introns",
                       task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []