from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: eucalyptus
====
Examples: 736
====
URL: https://www.openml.org/search?type=data&id=188
====
Description: **Author**: Bruce Bulloch    
**Source**: [WEKA Dataset Collection](http://www.cs.waikato.ac.nz/ml/weka/datasets.html) - part of the agridatasets archive. [This is the true source](http://tunedit.org/repo/Data/Agricultural/eucalyptus.arff)  
**Please cite**: None  

**Eucalyptus Soil Conservation**  
The objective was to determine which seedlots in a species are best for soil conservation in seasonally dry hill country. Determination is found by measurement of height, diameter by height, survival, and other contributing factors. 
 
It is important to note that eucalypt trial methods changed over time; earlier trials included mostly 15 - 30cm tall seedling grown in peat plots and the later trials have included mostly three replications of eight trees grown. This change may contribute to less significant results.

Experimental data recording procedures which require noting include:
 - instances with no data recorded due to experimental recording procedures
   require that the absence of a species from one replicate at a site was
   treated as a missing value, but if absent from two or more replicates at a
   site the species was excluded from the site's analyses.
 - missing data for survival, vigour, insect resistance, stem form, crown form
   and utility especially for the data recorded at the Morea Station; this 
   could indicate the death of species in these areas or a lack in collection
   of data.  

### Attribute Information  
 
  1.  Abbrev - site abbreviation - enumerated
  2.  Rep - site rep - integer
  3.  Locality - site locality in the North Island - enumerated
  4.  Map_Ref - map location in the North Island - enumerated
  5.  Latitude - latitude approximation - enumerated
  6.  Altitude - altitude approximation - integer
  7.  Rainfall - rainfall (mm pa) - integer
  8.  Frosts - frosts (deg. c) - integer
  9.  Year - year of planting - integer
  10. Sp - species code - enumerated
  11. PMCno - seedlot number - integer
  12. DBH - best diameter base height (cm) - real
  13. Ht - height (m) - real
  14. Surv - survival - integer
  15. Vig - vigour - real
  16. Ins_res - insect resistance - real
  17. Stem_Fm - stem form - real
  18. Crown_Fm - crown form - real
  19. Brnch_Fm - branch form - real
  Class:
  20. Utility - utility rating - enumerated

### Relevant papers

Bulluch B. T., (1992) Eucalyptus Species Selection for Soil Conservation in Seasonally Dry Hill Country - Twelfth Year Assessment  New Zealand Journal of Forestry Science 21(1): 10 - 31 (1991)  

Kirsten Thomson and Robert J. McQueen (1996) Machine Learning Applied to Fourteen Agricultural Datasets. University of Waikato Research Report  
https://www.cs.waikato.ac.nz/ml/publications/1996/Thomson-McQueen-96.pdf + the original publication:
====
Target Variable: Utility (nominal, 5 distinct): ['good', 'none', 'average', 'low', 'best']
====
Features:

Abbrev (nominal, 16 distinct): ['Puk', 'Wak', 'Wai', 'K81', 'Mor', 'WSp', 'Paw', 'Lon', 'K83', 'K82']
Rep (numeric, 4 distinct): ['1', '3', '2', '22']
Locality (nominal, 8 distinct): ['South_Wairarapa', 'Central_Wairarapa', 'Central_Hawkes_Bay', 'Southern_Hawkes_Bay', 'Central_Hawkes_Bay_(coastal)', 'Southern_Hawkes_Bay_(coastal)', 'Northern_Hawkes_Bay', 'Central_Poverty_Bay']
Map_Ref (nominal, 14 distinct): ['N158_344/626', 'N166_063/197', 'N162_081/300', 'N142_377/957', 'N141_295/063', 'N151_912/221', 'N146_273/737', 'N162_097/424', 'N158_343/625', 'N135_382/137']
Latitude (nominal, 12 distinct): ['40__57', '41__16', '41__12', '39__50', '40__36', '39__43', '40__00', '41__08', '39__38', '39__00']
Altitude (numeric, 9 distinct): ['180.0', '150.0', '300.0', '160.0', '70.0', '220.0', '100.0', '200.0', '130.0']
Rainfall (numeric, 10 distinct): ['1080.0', '1000.0', '1200.0', '1250.0', '900.0', '1300.0', '1050.0', '850.0', '1400.0', '1750.0']
Frosts (numeric, 2 distinct): ['-3.0', '-2.0']
Year (numeric, 5 distinct): ['1981.0', '1983.0', '1982.0', '1980.0', '1986.0']
Sp (nominal, 27 distinct): ['nd', 're', 'ov', 'fr', 'fa', 'ob', 'am', 'pu', 'rd', 'ni']
PMCno (numeric, 86 distinct): ['1596.0', '2548.0', '2575.0', '1482.0', '1522.0', '1524.0', '2571.0', '2569.0', '2553.0', '1111.0']
DBH (numeric, 604 distinct): ['19.6', '20.8', '15.0', '17.0', '10.4', '17.6', '7.8', '10.5', '6.97', '11.9']
Ht (numeric, 532 distinct): ['10.0', '5.6', '10.2', '5.5', '7.0', '12.3', '9.9', '10.8', '11.8', '9.0']
Surv (numeric, 48 distinct): ['100.0', '75.0', '50.0', '88.0', '63.0', '25.0', '13.0', '38.0', '10.0', '40.0']
Vig (numeric, 34 distinct): ['3.0', '4.0', '2.0', '3.5', '2.5', '1.0', '5.0', '4.5', '1.5', '3.3']
Ins_res (numeric, 29 distinct): ['3.0', '4.0', '2.0', '3.5', '2.5', '1.5', '1.0', '2.7', '4.5', '1.8']
Stem_Fm (numeric, 27 distinct): ['3.0', '2.0', '4.0', '3.5', '2.5', '3.3', '3.2', '1.0', '2.8', '5.0']
Crown_Fm (numeric, 30 distinct): ['4.0', '3.0', '3.5', '2.0', '2.5', '3.3', '4.5', '1.0', '3.7', '1.5']
Brnch_Fm (numeric, 29 distinct): ['3.0', '2.0', '4.0', '3.5', '2.5', '1.0', '1.5', '3.3', '2.7', '2.8']
'''

CONTEXT = "Eucalyptus Soil Conservation"
TARGET = CuratedTarget(raw_name="Utility", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []
