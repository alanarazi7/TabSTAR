from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: anneal
====
Examples: 898
====
URL: https://www.openml.org/search?type=data&id=2
====
Description: **Author**: Unknown. Donated by David Sterling and Wray Buntine  

**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Annealing) - 1990  

**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  



The original Annealing dataset from UCI. The exact meaning of the features and classes is largely unknown. Annealing, in metallurgy and materials science, is a heat treatment that alters the physical and sometimes chemical properties of a material to increase its ductility and reduce its hardness, making it more workable. It involves heating a material to above its recrystallization temperature, maintaining a suitable temperature, and then cooling. (Wikipedia)



### Attribute Information:

     1. family:          --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS

     2. product-type:    C, H, G

     3. steel:           -,R,A,U,K,M,S,W,V

     4. carbon:          continuous

     5. hardness:        continuous

     6. temper_rolling:  -,T

     7. condition:       -,S,A,X

     8. formability:     -,1,2,3,4,5

     9. strength:        continuous

    10. non-ageing:      -,N

    11. surface-finish:  P,M,-

    12. surface-quality: -,D,E,F,G

    13. enamelability:   -,1,2,3,4,5

    14. bc:              Y,-

    15. bf:              Y,-

    16. bt:              Y,-

    17. bw/me:           B,M,-

    18. bl:              Y,-

    19. m:               Y,-

    20. chrom:           C,-

    21. phos:            P,-

    22. cbond:           Y,-

    23. marvi:           Y,-

    24. exptl:           Y,-

    25. ferro:           Y,-

    26. corr:            Y,-

    27. blue/bright/varn/clean:          B,R,V,C,-

    28. lustre:          Y,-

    29. jurofm:          Y,-

    30. s:               Y,-

    31. p:               Y,-

    32. shape:           COIL, SHEET

    33. thick:           continuous

    34. width:           continuous

    35. len:             continuous

    36. oil:             -,Y,N

    37. bore:            0000,0500,0600,0760

    38. packing: -,1,2,3

    classes:        1,2,3,4,5,U

  

    -- The '-' values are actually 'not_applicable' values rather than

       'missing_values' (and so can be treated as legal discrete

       values rather than as showing the absence of a discrete value).
====
Target Variable: class (nominal, 5 distinct): ['3', '2', '5', 'U', '1', '4']
====
Features:

family (nominal, 3 distinct): ['TN', 'ZS', 'GB', 'GK', 'GS', 'ZA', 'ZF', 'ZH', 'ZM']
product-type (nominal, 1 distinct): ['C', 'H', 'G']
steel (nominal, 8 distinct): ['A', 'R', 'K', 'M', 'W', 'V', 'S', 'U']
carbon (numeric, 10 distinct): ['0.0', '55.0', '45.0', '65.0', '6.0', '70.0', '4.0', '8.0', '10.0', '3.0']
hardness (numeric, 7 distinct): ['0.0', '45.0', '85.0', '50.0', '60.0', '70.0', '80.0']
temper_rolling (nominal, 2 distinct): ['T']
condition (nominal, 3 distinct): ['S', 'A', 'X']
formability (nominal, 5 distinct): ['2', '3', '1', '5', '4']
strength (numeric, 8 distinct): ['0.0', '310.0', '500.0', '600.0', '350.0', '400.0', '300.0', '700.0']
non-ageing (nominal, 2 distinct): ['N']
surface-finish (nominal, 2 distinct): ['P', 'M']
surface-quality (nominal, 5 distinct): ['E', 'G', 'F', 'D']
enamelability (nominal, 3 distinct): ['2', '1', '3', '4', '5']
bc (nominal, 2 distinct): ['Y']
bf (nominal, 2 distinct): ['Y']
bt (nominal, 2 distinct): ['Y']
bw%2Fme (nominal, 3 distinct): ['B', 'M']
bl (nominal, 2 distinct): ['Y']
m (nominal, 1 distinct): ['Y']
chrom (nominal, 2 distinct): ['C']
phos (nominal, 2 distinct): ['P']
cbond (nominal, 2 distinct): ['Y']
marvi (nominal, 1 distinct): ['Y']
exptl (nominal, 2 distinct): ['Y']
ferro (nominal, 2 distinct): ['Y']
corr (nominal, 1 distinct): ['Y']
blue%2Fbright%2Fvarn%2Fclean (nominal, 4 distinct): ['B', 'V', 'C', 'R']
lustre (nominal, 2 distinct): ['Y']
jurofm (nominal, 1 distinct): ['Y']
s (nominal, 1 distinct): ['Y']
p (nominal, 1 distinct): ['Y']
shape (nominal, 2 distinct): ['SHEET', 'COIL']
thick (numeric, 50 distinct): ['0.7', '1.6', '0.699', '0.6', '0.8', '3.2', '0.3', '1.2', '0.4', '1.599']
width (numeric, 68 distinct): ['610.0', '1320.0', '609.9', '1220.0', '1300.0', '900.0', '20.0', '50.0', '1250.0', '150.0']
len (numeric, 24 distinct): ['0.0', '762.0', '4880.0', '612.0', '4170.0', '761.0', '3000.0', '301.0', '150.0', '1.0']
oil (nominal, 3 distinct): ['Y', 'N']
bore (nominal, 3 distinct): ['0', '600', '500', '760']
packing (nominal, 3 distinct): ['3', '2', '1']
'''

CONTEXT = "Anneal Chemical Process"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []