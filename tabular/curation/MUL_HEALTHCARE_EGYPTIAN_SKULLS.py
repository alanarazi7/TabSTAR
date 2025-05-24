from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: EgyptianSkulls
====
Examples: 150
====
URL: https://www.openml.org/search?type=data&id=1099
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Datasets of Data And Story Library, project illustrating use of basic statistic methods, converted to arff format by Hakan Kjellerstrand.
Source: TunedIT: http://tunedit.org/repo/DASL

DASL file http://lib.stat.cmu.edu/DASL/Datafiles/EgyptianSkulls.html

Egyptian Skull Development

Reference:   Thomson, A. and Randall-Maciver, R. (1905) Ancient Races of the Thebaid, Oxford:  Oxford University Press.
Also found in:  Hand, D.J., et al. (1994) A Handbook of Small Data Sets, New York:  Chapman & Hall, pp. 299-301.
Manly, B.F.J. (1986) Multivariate Statistical Methods, New York:  Chapman & Hall.
Authorization:   Contact Authors
Description:   Four measurements of male Egyptian skulls from 5 different time periods.  Thirty skulls are measured from each time period.


Number of cases:   150
Variable Names:

MB:   Maximal Breadth of Skull
BH:   Basibregmatic Height of Skull
BL:   Basialveolar Length of Skull
NH:   Nasal Height of Skull
Year:   Approximate Year of Skull Formation (negative = B.C., positive = A.D.)
====
Target Variable: Year (numeric, 5 distinct): ['-4000.0', '-3300.0', '-1850.0', '-200.0', '150.0']
====
Features:

MB (numeric, 26 distinct): ['131', '138', '136', '132', '134', '137', '133', '135', '130', '126']
BH (numeric, 24 distinct): ['134', '136', '135', '131', '130', '133', '138', '129', '137', '132']
BL (numeric, 27 distinct): ['95', '99', '100', '97', '93', '101', '92', '96', '98', '91']
NH (numeric, 16 distinct): ['50', '51', '53', '54', '52', '48', '49', '47', '55', '46']
'''

CONTEXT = "Egyptian Skulls from different periods"
TARGET = CuratedTarget(raw_name="Year", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="MB", new_name="Maximal Breadth of Skull"),
            CuratedFeature(raw_name="BH", new_name="Basibregmatic Height of Skull"),
            CuratedFeature(raw_name="BL", new_name="Basialveolar Length of Skull"),
            CuratedFeature(raw_name="NH", new_name="Nasal Height of Skull")]