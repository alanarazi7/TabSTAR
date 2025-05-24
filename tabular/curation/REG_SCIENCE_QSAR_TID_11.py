from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Description
Author: Dr Ivan Olier, Dr Jeremy Besnard, Dr Noureddin Sadawi, Dr Larisa Soldatova, Dr Crina Grosan, Prof Ross King, Dr Richard Bickerton, Prof Andrew Hopkins and Dr Willem van Hoorn
Source: MetaQSAR project - September 2015
Please cite:

This dataset contains QSAR data (from ChEMBL version 17) showing activity values (unit is pseudo-pCI50) of several compounds on drug target TID: 11, and it has 5742 rows and 1026 features (including IDs and class feature: MOLECULE_CHEMBL_ID and MEDIAN_PXC50). The features represent FCFP 1024bit Molecular Fingerprints which were generated from SMILES strings. They were obtained using the Pipeline Pilot program, Dassault Systèmes BIOVIA. Generating Fingerprints does not usually require missing value imputation as all bits are generated.
'''

CONTEXT = "QSAR TID 11 - Molecular Fingerprints"
TARGET = CuratedTarget(raw_name="MEDIAN_PXC50", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []