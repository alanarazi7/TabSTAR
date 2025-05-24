from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: physiochemical_protein
====
Examples: 45730
====
URL: https://www.openml.org/search?type=data&id=44963
====
Description: **Data Description**

This is a data set of Physicochemical Properties of Protein Tertiary Structure. The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

The goal of the dataset is to predict the size of the residue for a tertiary protein structure (a 3d protein structure). Once linked in the protein chain, an individual amino acid is called a residue. The target feature is root mean square error of the residue.

**Attribute Description**

1. *RMSD* - size of the residue
2. *F1* - total surface area
3. *F2* - non polar exposed area
4. *F3* - fractional area of exposed non polar residue
5. *F4* - fractional area of exposed non polar part of residue
6. *F5* - molecular mass weighted exposed area
7. *F6* - average deviation from standard exposed area of residue
8. *F7* - Euclidian distance
9. *F8* - secondary structure penalty
10. *F9* - Spacial Distribution constraints (N,K Value)
====
Target Variable: RMSD (numeric, 15903 distinct): ['0.0', '1.787', '2.006', '1.9', '2.055', '1.811', '1.896', '2.527', '1.932', '1.937']
====
Features:

F1 (numeric, 39916 distinct): ['13475.4', '5811.82', '4000.26', '14170.5', '20734.4', '4670.89', '15024.1', '6001.6', '10807.5', '10504.3']
F2 (numeric, 39863 distinct): ['4814.93', '1087.13', '1053.23', '7997.71', '2129.8', '1866.32', '3520.75', '3102.87', '1729.67', '2067.07']
F3 (numeric, 20089 distinct): ['0.3573', '0.2485', '0.2718', '0.3118', '0.2663', '0.1812', '0.2873', '0.269', '0.2795', '0.2807']
F4 (numeric, 40374 distinct): ['168.55', '33.732', '52.5591', '186.407', '46.7282', '189.396', '174.306', '191.592', '56.5611', '49.8648']
F5 (numeric, 41868 distinct): ['1877843.5474', '569494.3892', '799977.1539', '1937925.8075', '686984.9356', '2876946.082', '659783.0916', '952969.4869', '537119.5301', '820445.0505']
F6 (numeric, 39155 distinct): ['227.605', '46.039', '65.2332', '214.666', '66.6405', '356.061', '118.232', '211.148', '133.842', '128.604']
F7 (numeric, 39450 distinct): ['4644.75', '4057.08', '1773.46', '3034.98', '1399.62', '4581.39', '2334.29', '4628.02', '3903.68', '3232.54']
F8 (numeric, 341 distinct): ['32.0', '17.0', '40.0', '36.0', '30.0', '41.0', '33.0', '27.0', '39.0', '38.0']
F9 (numeric, 37299 distinct): ['46.5464', '29.7563', '34.8833', '39.7659', '44.4892', '38.8321', '44.4197', '38.1176', '44.2712', '28.9962']
'''

CONTEXT = "Physiochemical Protein for Tertiary Structure"
TARGET = CuratedTarget(raw_name="RMSD", new_name="Size of the Residue", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="F1", new_name="Total Surface Area"),
            CuratedFeature(raw_name="F2", new_name="Non Polar Exposed Area"),
            CuratedFeature(raw_name="F3", new_name="Fractional Area of Exposed Non Polar Residue"),
            CuratedFeature(raw_name="F4", new_name="Fractional Area of Exposed Non Polar Part of Residue"),
            CuratedFeature(raw_name="F5", new_name="Molecular Mass Weighted Exposed Area"),
            CuratedFeature(raw_name="F6", new_name="Average Deviation from Standard Exposed Area of Residue"),
            CuratedFeature(raw_name="F7", new_name="Euclidian Distance"),
            CuratedFeature(raw_name="F8", new_name="Secondary Structure Penalty"),
            CuratedFeature(raw_name="F9", new_name="Spacial Distribution Constraints (N, K Value)")]