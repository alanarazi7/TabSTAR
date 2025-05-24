from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: yeast
====
Examples: 1484
====
URL: https://www.openml.org/search?type=data&id=181
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:
====
Target Variable: class_protein_localization (nominal, 10 distinct): ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL']
====
Features:

mcg (numeric, 81 distinct): ['0.45', '0.46', '0.47', '0.49', '0.5', '0.51', '0.48', '0.43', '0.4', '0.58']
gvh (numeric, 79 distinct): ['0.46', '0.51', '0.48', '0.45', '0.44', '0.5', '0.53', '0.49', '0.56', '0.43']
alm (numeric, 53 distinct): ['0.53', '0.54', '0.52', '0.5', '0.51', '0.49', '0.56', '0.55', '0.57', '0.47']
mit (numeric, 78 distinct): ['0.18', '0.16', '0.19', '0.17', '0.15', '0.22', '0.21', '0.13', '0.14', '0.2']
erl (numeric, 2 distinct): ['0.5', '1.0']
pox (numeric, 3 distinct): ['0.0', '0.83', '0.5']
vac (numeric, 48 distinct): ['0.51', '0.52', '0.53', '0.5', '0.49', '0.54', '0.48', '0.55', '0.47', '0.46']
nuc (numeric, 68 distinct): ['0.22', '0.27', '0.25', '0.31', '0.26', '0.28', '0.33', '0.32', '0.3', '0.11']
'''

CONTEXT = "Yeast Protein Localization"
TARGET = CuratedTarget(raw_name="class_protein_localization", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"CYT": "Cytoplasm", "NUC": "Nucleus", "MIT": "Mitochondria",
                                      "ME3": "Extracellular Membrane", "ME2": "Cytoplasmic Membrane",
                                      "ME1": "Nuclear Membrane", "EXC": "Extracellular", "VAC": "Vacuole",
                                      "POX": "Peroxisome", "ERL": "Endoplasmic Reticulum"})
COLS_TO_DROP = []
FEATURES = []
