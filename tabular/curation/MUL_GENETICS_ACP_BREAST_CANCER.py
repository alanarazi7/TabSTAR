from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: acp-breast-cancer
====
Examples: 949
====
URL: https://www.openml.org/search?type=data&id=42895
====
Description: Author: Francesca Grisoni, Claudia S. Neuhaus, Miyabi Hishinuma, Gisela Gabernet, Jan A. Hiss, - Masaaki Kotera, Gisbert Schneider
Source: [UCI](https://archive.ics.uci.edu/ml/datasets/Anticancer+peptides) - 2019
Please cite: [Paper](https://link.springer.com/article/10.1007/s00894-019-4007-6)

Peptides with experimental annotations on their anticancer action on breast and lung cancer cells. This dataset only contains the breast cancer data.

Membranolytic anticancer peptides (ACPs) are drawing increasing attention as potential future therapeutics against cancer, due to their ability to hinder the development of cellular resistance and their potential to overcome common hurdles of chemotherapy, e.g., side effects and cytotoxicity.
This dataset contains information on peptides (annotated for their one-letter amino acid code) and their anticancer activity on breast and lung cancer cell lines. The final training sets contained 949 peptides for Breast cancer and 901 peptides for Lung cancer.

### Attribute Information:

The dataset contains three attributes:
1. Peptide ID
2. One-letter amino-acid sequence
3. Class (active, moderately active, experimental inactive, virtual inactive)
====
Features:

ID (numeric, 949 distinct): ['1', '639', '627', '628', '629', '630', '631', '632', '633', '634']
sequence (string, 949 distinct): ['AAWKWAWAKKWAKAKKWAKAA', 'PAERYYRDARIT', 'NSSDSIDWLTSM', 'NSVLRAVAEVYA', 'NTDTLERVTEIFKALG', 'NTEKLLKTVPIIQNQMDALLD', 'NTPEAYSVLFDMAREVNRLKAED', 'NVGKAWAEDVLALVKH', 'NVKDLADAAKRT', 'NVKDVTKLVAN']
class (string, 4 distinct): ['inactive - virtual', 'mod. active', 'inactive - exp', 'very active']
'''

CONTEXT = "ACP Breast Cancer Gene Expression Dataset"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["ID"]
FEATURES = []