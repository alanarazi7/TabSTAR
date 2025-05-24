from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: QSAR_fish_toxicity
====
Examples: 908
====
URL: https://www.openml.org/search?type=data&id=44970
====
Description: **Data Description**

Data set containing values for 6 attributes (molecular descriptors) of 908 chemicals used to predict quantitative acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow).

This dataset was used to develop quantitative regression QSAR models to predict acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow) on a set of 908 chemicals. LC50 data, which is the concentration that causes death in 50% of test fish over a test duration of 96 hours, was used as model response.

**Attribute Description**

The model comprised 6 molecular descriptors

1. *CIC0* - information indices
2. *SM1_Dz* - 2D matrix-based descriptors
3. *GATS1i* - 2D autocorrelations
4. *NdsCH* - atom-type counts
5. *NdssC* - atom-type counts
6. *MLOGP* - molecular properties
7. *LC50* - quantitative response, LC50 [-LOG(mol/L)], target feature
====
Target Variable: LC50 (numeric, 827 distinct): ['3.513', '4.208', '3.979', '3.926', '3.66', '3.47', '3.92', '4.577', '4.691', '1.842']
====
Features:

CIC0 (numeric, 502 distinct): ['2.126', '3.08', '2.377', '2.479', '2.508', '2.08', '2.834', '3.252', '2.233', '3.739']
SM1_Dz (numeric, 186 distinct): ['0.223', '0.134', '0.405', '0.331', '0.0', '0.693', '0.56', '0.496', '0.251', '0.83']
GATS1i (numeric, 557 distinct): ['0.941', '1.179', '0.871', '0.954', '0.938', '1.189', '1.571', '1.6', '1.288', '1.077']
NdsCH (numeric, 5 distinct): ['0', '1', '2', '4', '3']
NdssC (numeric, 7 distinct): ['0', '1', '2', '3', '4', '6', '5']
MLOGP (numeric, 559 distinct): ['0.8', '1.701', '2.604', '0.202', '1.748', '1.064', '2.193', '1.587', '1.442', '3.291']
'''

CONTEXT = "Fish Toxicity Prediction"
TARGET = CuratedTarget(raw_name="LC50", new_name="LC50 Quantitative Response", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="CIC0", new_name="CIC0 - Information Indices"),
            CuratedFeature(raw_name="SM1_Dz", new_name="SM1_Dz - 2D Matrix-based Descriptors"),
            CuratedFeature(raw_name="GATS1i", new_name="GATS1i - 2D Autocorrelations"),
            CuratedFeature(raw_name="NdsCH", new_name="NdsCH - Atom-type Counts"),
            CuratedFeature(raw_name="NdssC", new_name="NdssC - Atom-type Counts"),
            CuratedFeature(raw_name="MLOGP", new_name="MLOGP - Molecular Properties")]