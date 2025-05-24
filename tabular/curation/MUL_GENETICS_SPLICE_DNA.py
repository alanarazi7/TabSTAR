from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: splice
====
Examples: 3190
====
URL: https://www.openml.org/search?type=data&id=46
====
Description: **Author**: Genbank. Donated by G. Towell, M. Noordewier, and J. Shavlik  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences))   
**Please cite**:  None  

Primate splice-junction gene sequences (DNA) with associated imperfect domain theory.
Splice junctions are points on a DNA sequence at which 'superfluous' DNA is removed during the process of protein creation in higher organisms. The problem posed in this dataset is to recognize, given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing) and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks: recognizing exon/intron boundaries (referred to as EI sites), and recognizing intron/exon boundaries (IE sites). (In the biological community, IE borders are referred to a ''acceptors'' while EI borders are referred to as ''donors''.)

All examples taken from Genbank 64.1. Categories "ei" and "ie" include every "split-gene" for primates in Genbank 64.1. Non-splice examples taken from sequences known not to include a splicing site.
         
### Attribute Information 
>
              1   One of {n ei ie}, indicating the class.
              2   The instance name.
           3-62   The remaining 60 fields are the sequence, starting at 
                  position -30 and ending at position +30. Each of
                  these fields is almost always filled by one of 
                  {a, g, t, c}. Other characters indicate ambiguity among
                  the standard characters according to the following table:
    character: meaning
        D: A or G or T
        N: A or G or C or T
        S: C or G
        R: A or G

Notes:  
* Instance_name is an identifier and should be ignored for modelling
====
Target Variable: Class (nominal, 3 distinct): ['N', 'IE', 'EI']
====
Features:

attribute_1 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'D']
attribute_2 (nominal, 5 distinct): ['C', 'G', 'A', 'T', 'D']
attribute_3 (nominal, 4 distinct): ['C', 'G', 'T', 'A']
attribute_4 (nominal, 4 distinct): ['C', 'G', 'A', 'T']
attribute_5 (nominal, 4 distinct): ['C', 'T', 'A', 'G']
attribute_6 (nominal, 4 distinct): ['C', 'G', 'T', 'A']
attribute_7 (nominal, 4 distinct): ['C', 'A', 'G', 'T']
attribute_8 (nominal, 4 distinct): ['C', 'T', 'A', 'G']
attribute_9 (nominal, 4 distinct): ['C', 'T', 'A', 'G']
attribute_10 (nominal, 4 distinct): ['T', 'C', 'A', 'G']
attribute_11 (nominal, 4 distinct): ['T', 'C', 'G', 'A']
attribute_12 (nominal, 4 distinct): ['C', 'G', 'T', 'A']
attribute_13 (nominal, 4 distinct): ['C', 'T', 'G', 'A']
attribute_14 (nominal, 5 distinct): ['C', 'A', 'T', 'G', 'N']
attribute_15 (nominal, 4 distinct): ['C', 'T', 'G', 'A']
attribute_16 (nominal, 4 distinct): ['C', 'T', 'A', 'G']
attribute_17 (nominal, 4 distinct): ['T', 'C', 'G', 'A']
attribute_18 (nominal, 4 distinct): ['T', 'C', 'G', 'A']
attribute_19 (nominal, 5 distinct): ['C', 'T', 'G', 'A', 'N']
attribute_20 (nominal, 5 distinct): ['T', 'C', 'A', 'G', 'N']
attribute_21 (nominal, 5 distinct): ['C', 'T', 'G', 'A', 'N']
attribute_22 (nominal, 5 distinct): ['C', 'T', 'G', 'A', 'N']
attribute_23 (nominal, 5 distinct): ['C', 'T', 'G', 'A', 'N']
attribute_24 (nominal, 5 distinct): ['C', 'T', 'G', 'A', 'N']
attribute_25 (nominal, 5 distinct): ['C', 'T', 'G', 'A', 'N']
attribute_26 (nominal, 5 distinct): ['T', 'C', 'A', 'G', 'N']
attribute_27 (nominal, 5 distinct): ['C', 'G', 'A', 'T', 'N']
attribute_28 (nominal, 5 distinct): ['C', 'A', 'T', 'G', 'N']
attribute_29 (nominal, 5 distinct): ['A', 'C', 'G', 'T', 'N']
attribute_30 (nominal, 5 distinct): ['G', 'A', 'T', 'C', 'N']
attribute_31 (nominal, 5 distinct): ['G', 'A', 'C', 'T', 'N']
attribute_32 (nominal, 5 distinct): ['T', 'A', 'C', 'G', 'N']
attribute_33 (nominal, 5 distinct): ['A', 'G', 'C', 'T', 'N']
attribute_34 (nominal, 5 distinct): ['A', 'G', 'C', 'T', 'N']
attribute_35 (nominal, 6 distinct): ['G', 'C', 'T', 'A', 'N', 'R']
attribute_36 (nominal, 6 distinct): ['T', 'C', 'G', 'A', 'N', 'S']
attribute_37 (nominal, 5 distinct): ['G', 'A', 'C', 'T', 'N']
attribute_38 (nominal, 5 distinct): ['C', 'G', 'A', 'T', 'N']
attribute_39 (nominal, 5 distinct): ['C', 'G', 'T', 'A', 'N']
attribute_40 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_41 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_42 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_43 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'N']
attribute_44 (nominal, 5 distinct): ['C', 'G', 'T', 'A', 'N']
attribute_45 (nominal, 5 distinct): ['C', 'G', 'A', 'T', 'N']
attribute_46 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'N']
attribute_47 (nominal, 5 distinct): ['G', 'T', 'C', 'A', 'N']
attribute_48 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'N']
attribute_49 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'N']
attribute_50 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'N']
attribute_51 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_52 (nominal, 5 distinct): ['G', 'A', 'C', 'T', 'N']
attribute_53 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_54 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_55 (nominal, 5 distinct): ['G', 'C', 'A', 'T', 'N']
attribute_56 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_57 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
attribute_58 (nominal, 5 distinct): ['C', 'G', 'T', 'A', 'N']
attribute_59 (nominal, 5 distinct): ['C', 'G', 'A', 'T', 'N']
attribute_60 (nominal, 5 distinct): ['G', 'C', 'T', 'A', 'N']
'''

GENE_MAPPING = {"D": "A or G or T",
                "N": "A or G or C or T",
                "S": "C or G",
                "R": "A or G"}

CONTEXT = "Genetics primate splice-junction gene sequences (DNA)"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=f"attribute_{i}",
                           new_name=f"Sequence Position {i-31 if i < 31 else i-30}",
                           value_mapping=GENE_MAPPING,
                           allow_missing_key=True) for i in range(1, 61)]