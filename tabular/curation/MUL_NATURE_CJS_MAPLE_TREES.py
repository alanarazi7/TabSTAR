from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cjs
====
Examples: 2796
====
URL: https://www.openml.org/search?type=data&id=23380
====
Description: **Author**: Dr. Fernando Camacho  
**Source**: Unknown - 1995  
**Please cite**: Camacho, F. and Arron, G. (1995)  Effects of the regulators paclobutrazol and flurprimidol on the growth of terminal sprouts formed on trimmed silver maple trees. Canadian Journal of Statistics 3(23).

Data on tree growth used in the Case Study published in the September, 1995 issue of the Canadian Journal of Statistics. This data set was been provided by Dr. Fernando Camacho, Ontario Hydro Technologies, 800 Kipling Ave, Toronto Canada M3Z 5S4. It forms the basis of the Case Study in Data Analysis published in the Canadian Journal of Statistics, September 1995. It can be freely used for noncommercial purposes, as long as proper acknowledgement to the source and to the Canadian Journal of Statistics is made.


Description


The effects of the Growth Regulators Paclobutrazol (PP 333)
and Flurprimidol (EL-500) on the Number and Length of Internodes
in Terminal Sprouts Formed on Trimmed Silver Maple Trees.
 
Introduction:
 
The trimming of trees under distribution lines on city streets and
in rural areas is a major problem and expense for electrical
utilities.  Such operations are routinely performed at intervals of
one to eight years depending upon the individual species growth rate
and the amount of clearance required.  Ontario Hydro trims about
500,000 trees per year at a cost of about $25 per tree.
 
Much effort has been spent in developing chemicals for the horticultural
industry to retard the growth of woody and herbaceous plants.  Recently,
a group of new growth regulators was introduced which was shown to be
effective in controlling the growth of trees without producing
noticeable injury symptoms.  In this group are PP 333 ( common name
paclobutrazol) (2RS, 3RS - 1 -(4-chlorophenyl) - 4,4 - dimethyl - 2 -
(1,2,4-triazol-l-yl) pentan - 3- ol and EL-500 (common name flurprimidol
and composition alpha - (1-methylethyl) - alpha - [4-(trifluromethoxyl)
phenyl] - 5- pyrimidine - methanol).  Both EL-500 and PP-333 have been
reported to control excessive sprout growth in a number of species
when applied as a foliar spray, as a soil drench, or by trunk injection.
Sprout length is a function of both the number of internodes and
the length of the individual internodes in the sprout.  While there
have been many reports that both PP 333 and EL-500 cause a reduction
in the length of internodes formed in sprouts on woody plants treated
with the growth regulators, there has been but one report that EL-500
application to apple trees resulted in a reduction of the number
of internodes formed per sprout.
 
The purpose of the present study was to investigate the length of the
terminal sprouts, the length of the individual internodes in those
sprouts, and the number of internodes in trimmed silver maple trees
following trunk injection with the growth regulators PP 333 and EL-500.
 
Experimental Details.
 
Multistemmed 12-year-old silver maple trees growing at Wesleyville,
Ontario were trunk injected with methanolic solutions of EL-500
and PP-333 in May of 1985 using a third generation Asplundh
tree injector.
 
Two different application rates (20 g/L and 4 g/L) were used for each
chemical.  The volume of solution (and hence the amount of active
ingredient) injected into each tree was determined from the diameter
of the tree, using the formula: vol(mL) = (dbh)*(dbh)*.492 where dbh
is the diameter at breast height.  Two sets of control trees were
included in the experiment.  In one set, tree received no injection
(control) and in a second set, the trees were injected with
methanol, the carrier in the growth regulator solutions.  Ten trees,
chosen at random, were used in each of the control and experimental
sets.  Prior to injection, all the trees were trimmed by a forestry
crew, with their heights being reduced by about one third.
 
In January 1987, twenty months after the trees were injected, between
six and eight limbs were removed at random from the bottom two-thirds
of the canopy of each of the ten trees in each experimental and control
set.  The limbs were returned to the laboratory and the length of all
the terminal sprouts, the lengths of the individual internodes, and
the number of internodes recorded.  Between one and 25 terminal
sprouts were found on each limb collected.  Sprouts which had a
length of 1 cm or less were recorded as being 1 cm in length.
In such spouts, the internode lengths were not measured, but were
calculated from the total length of the sprout and the number
of internodes counted.  Internode lengths were then expressed to one
decimal place.  In two instances, one of the ten trees in a set
could not be sampled because limb removal would have jeopardized the
health of the tree over the long-term.
 
Data set:
 
Each of the records represents a terminal sprout and contains the
following information:
   N   the sprout number
   TR  treatment 1  control
                  2  methanol control
                  3  PP 333 20g/L
                  4  PP 333  4g/L
                  5  EL 500 20g/L
                  6  EL 500  4g/L
   TREE  tree id
   BR    branch id
   TL    total sprout length (cm)
   IN    number of internodes on the sprout
   INTER a list of the lengths of the internodes in the sprout,
          starting from the base of the sprout (129 entries)
 
Sprouts 1868 to 1879 do not have branch identification data.
====
Target Variable: TR (nominal, 6 distinct): ['PP_333_20g/L', 'EL_500_20g/L', 'PP_333_4g/L', 'control', 'EL_500_4g/L', 'methanol_control']
====
Features:

TREE (nominal, 57 distinct): ['G27', 'Q17', 'D13', 'D22', 'D18', 'Q23', 'Q25', 'J31', 'J27', 'D10']
BR (nominal, 11 distinct): ['F', 'E', 'B', 'D', 'C', 'G', 'A', 'H', 'I', 'J']
TL (numeric, 170 distinct): ['1.0', '2.0', '3.0', '4.0', '1.5', '6.0', '5.0', '0.5', '8.0', '7.0']
IN (numeric, 26 distinct): ['3', '2', '4', '5', '6', '7', '8', '1', '9', '10']
INTERNODE_1 (numeric, 30 distinct): ['0.3', '0.5', '0.2', '0.6', '1.0', '0.4', '0.7', '0.8', '1.2', '1.5']
INTERNODE_2 (numeric, 69 distinct): ['0.3', '0.5', '0.6', '0.7', '1.0', '0.8', '0.4', '1.2', '1.5', '1.1']
INTERNODE_3 (numeric, 118 distinct): ['0.3', '0.2', '0.5', '0.6', '0.4', '0.7', '0.8', '1.0', '1.2', '1.1']
INTERNODE_4 (numeric, 129 distinct): ['0.3', '0.2', '0.6', '0.5', '0.7', '0.4', '0.8', '2.5', '1.2', '1.1']
INTERNODE_5 (numeric, 114 distinct): ['0.2', '0.3', '0.5', '0.7', '1.1', '1.5', '2.8', '3.5', '0.8', '1.0']
INTERNODE_6 (numeric, 116 distinct): ['0.2', '0.3', '0.4', '1.2', '0.7', '2.0', '4.5', '1.0', '6.2', '1.7']
INTERNODE_7 (numeric, 100 distinct): ['0.2', '1.5', '0.3', '0.7', '1.8', '6.6', '4.0', '0.8', '2.3', '1.6']
INTERNODE_8 (numeric, 96 distinct): ['0.2', '0.3', '0.6', '0.8', '2.5', '0.7', '6.2', '4.5', '4.0', '3.0']
INTERNODE_9 (numeric, 87 distinct): ['0.2', '0.3', '3.1', '5.2', '3.6', '0.6', '5.0', '7.3', '3.8', '5.6']
INTERNODE_10 (numeric, 87 distinct): ['0.2', '0.3', '5.0', '4.7', '6.0', '6.3', '0.5', '6.2', '5.3', '4.0']
INTERNODE_11 (numeric, 86 distinct): ['0.2', '4.8', '5.4', '6.0', '6.6', '0.5', '5.8', '5.5', '0.3', '6.5']
INTERNODE_12 (numeric, 79 distinct): ['0.2', '0.3', '5.0', '0.4', '6.0', '5.1', '6.6', '4.6', '5.6', '5.5']
INTERNODE_13 (numeric, 77 distinct): ['5.3', '0.2', '7.0', '4.0', '4.3', '0.5', '5.5', '3.8', '6.0', '3.5']
INTERNODE_14 (numeric, 73 distinct): ['6.0', '2.4', '5.8', '2.6', '4.6', '3.2', '4.0', '6.3', '2.5', '0.8']
INTERNODE_15 (numeric, 67 distinct): ['0.2', '4.3', '1.0', '0.8', '3.1', '4.5', '5.5', '4.1', '5.2', '2.9']
INTERNODE_16 (numeric, 53 distinct): ['0.2', '0.7', '0.5', '5.1', '0.6', '4.5', '0.3', '2.8', '3.1', '3.8']
INTERNODE_17 (numeric, 52 distinct): ['0.8', '3.5', '0.3', '1.9', '1.3', '3.4', '3.2', '0.5', '6.2', '2.5']
INTERNODE_18 (numeric, 44 distinct): ['0.3', '0.2', '1.8', '0.5', '0.8', '3.5', '2.6', '3.3', '0.4', '2.3']
INTERNODE_19 (numeric, 36 distinct): ['0.2', '0.3', '1.2', '0.8', '1.6', '1.0', '4.6', '2.8', '1.5', '3.6']
INTERNODE_20 (numeric, 24 distinct): ['0.3', '0.2', '0.8', '2.2', '1.2', '0.4', '0.5', '1.1', '1.4', '4.1']
INTERNODE_21 (numeric, 19 distinct): ['0.2', '0.3', '1.6', '2.9', '0.8', '1.3', '1.0', '1.7', '3.8', '0.7']
INTERNODE_22 (numeric, 11 distinct): ['2.6', '0.8', '0.6', '1.1', '0.4', '0.2', '1.2', '2.0', '0.5', '2.5']
INTERNODE_23 (numeric, 10 distinct): ['0.3', '0.6', '0.7', '0.5', '1.8', '2.6', '0.2', '1.5', '1.0']
INTERNODE_24 (numeric, 8 distinct): ['0.7', '0.3', '1.0', '0.2', '2.7', '0.6', '0.4']
INTERNODE_25 (numeric, 4 distinct): ['0.2', '0.3', '3.1']
INTERNODE_26 (numeric, 2 distinct): ['2.2']
INTERNODE_27 (numeric, 2 distinct): ['1.2']
INTERNODE_28 (numeric, 2 distinct): ['0.9']
INTERNODE_29 (numeric, 2 distinct): ['0.3']
'''

CONTEXT = "Effect of Growth Regulators on Maple Trees Growth"
TARGET = CuratedTarget(raw_name="TR", new_name="Treatment", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="TREE", new_name="TreeID"),
            CuratedFeature(raw_name="BR", new_name="BranchID"),
            CuratedFeature(raw_name="TL", new_name="Total Sprout Length"),
            CuratedFeature(raw_name="IN", new_name="Internode Count")]