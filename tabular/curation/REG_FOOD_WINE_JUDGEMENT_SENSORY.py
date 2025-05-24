from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: sensory
====
Examples: 576
====
URL: https://www.openml.org/search?type=data&id=546
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Data for the sensory evaluation experiment in Brien, C.J. and Payne,
R.W. (1996) Tiers, structure formulae and the analysis of complicated
experiments.  submitted for publication.
The experiment involved two phases.  In the field phase a viticultural
experiment was conducted to investigate the differences between 4
types of trellising and 2 methods of pruning.  The design was a
split-plot design in which the trellis types were assigned to the main
plots using two adjacent Youden squares of 3 rows and 4 columns.  Each
main plot was split into two subplots (or halfplots) and the methods
of pruning assigned at random independently to the two halfplots in
each main plot.  The produce of each halfplot was made into a wine so
that there were 24 wines altogether.
The second phase was an evaluation phase in which the produce from the
halplots was evaluated by 6 judges all of whom took part in 24
sittings.  In the first 12 sittings the judges evaluated the wines
made from the halfplots of one square; the final 12 sittings were to
evaluate the wines from the other square.  At each sitting, each judge
assessed two glasses of wine from each of the halplots of one of the
main plots.  The main plots allocated to the judges at each sitting
were determined as follows.  For the allocation of rows, each occasion
was subdivided into 3 intervals of 4 consecutive sittings.  During
each interval, each judge examined plots from one particular row,
these being determined using two 3x3 Latin squares for each occasion,
one for judges 1-3 and the other for judges 4-6.  At each sitting
judges 1-3 examined wines from one particular column and judges 4-6
examined wines from another column.  The columns were randomized to
the 2 sets of judges x 3 intervals x 4 sittings using duplicates of a
balanced incomplete block design for v=4 and k=2 that were latinized.
This balanced incomplete block design consists of three sets of 2
blocks, each set containing the 4 "treatments".  For each interval, a
different set of 2 blocks was taken and each block assigned to two
sittings, but with the columns within the block placed in reverse
order in one sitting compared to the other sitting.  Thus, in each
interval, a judge would evaluate a wine from each of the 4 columns.
The scores assigned in evaluating the wines, and the factors indexing
them, are given below.  The factors are as follows:
Occasion
Judges
Interval
Sittings
Position
Squares
Rows
Columns
Halfplot
Trellis
Method
followed by the response variable
Score
The scores are ordered so that the factors Occasion, Judges, Interval,
Sittings and Position are in standard order; the remaining factors are
in randomized order.


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: last
====
Target Variable: Score (numeric, 11 distinct): ['15.0', '15.5', '14.5', '16.0', '14.0', '16.5', '13.5', '17.0', '13.0', '17.5']
====
Features:

Occasion (nominal, 2 distinct): ['1', '2']
Judges (nominal, 6 distinct): ['1', '2', '3', '4', '5', '6']
Interval (nominal, 3 distinct): ['1', '2', '3']
Sittings (nominal, 4 distinct): ['1', '2', '3', '4']
Position (nominal, 4 distinct): ['1', '2', '3', '4']
Squares (nominal, 2 distinct): ['1', '2']
Rows (nominal, 3 distinct): ['1', '2', '3']
Columns (nominal, 4 distinct): ['1', '2', '3', '4']
Halfplot (nominal, 2 distinct): ['1', '2']
Trellis (nominal, 4 distinct): ['1', '2', '3', '4']
Method (nominal, 2 distinct): ['1', '2']
'''

CONTEXT = "Wine Quality Sensory Judgement"
TARGET = CuratedTarget(raw_name="Score", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []