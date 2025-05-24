from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: kr-vs-kp
====
Examples: 3196
====
URL: https://www.openml.org/search?type=data&id=3
====
Description: Author: Alen Shapiro
Source: [UCI](https://archive.ics.uci.edu/ml/datasets/Chess+(King-Rook+vs.+King-Pawn))
Please cite: [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)

1. Title: Chess End-Game -- King+Rook versus King+Pawn on a7
(usually abbreviated KRKPA7). The pawn on a7 means it is one square
away from queening. It is the King+Rook's side (white) to move.

2. Sources:
(a) Database originally generated and described by Alen Shapiro.
(b) Donor/Coder: Rob Holte (holte@uottawa.bitnet). The database
was supplied to Holte by Peter Clark of the Turing Institute
in Glasgow (pete@turing.ac.uk).
(c) Date: 1 August 1989

3. Past Usage:
- Alen D. Shapiro (1983,1987), "Structured Induction in Expert Systems",
Addison-Wesley. This book is based on Shapiro's Ph.D. thesis (1983)
at the University of Edinburgh entitled "The Role of Structured
Induction in Expert Systems".
- Stephen Muggleton (1987), "Structuring Knowledge by Asking Questions",
pp.218-229 in "Progress in Machine Learning", edited by I. Bratko
and Nada Lavrac, Sigma Press, Wilmslow, England SK9 5BB.
- Robert C. Holte, Liane Acker, and Bruce W. Porter (1989),
"Concept Learning and the Problem of Small Disjuncts",
Proceedings of IJCAI. Also available as technical report AI89-106,
Computer Sciences Department, University of Texas at Austin,
Austin, Texas 78712.

4. Relevant Information:
The dataset format is described below. Note: the format of this
database was modified on 2/26/90 to conform with the format of all
the other databases in the UCI repository of machine learning databases.

5. Number of Instances: 3196 total

6. Number of Attributes: 36

7. Attribute Summaries:
Classes (2): -- White-can-win ("won") and White-cannot-win ("nowin").
I believe that White is deemed to be unable to win if the Black pawn
can safely advance.
Attributes: see Shapiro's book.

8. Missing Attributes: -- none

9. Class Distribution:
In 1669 of the positions (52%), White can win.
In 1527 of the positions (48%), White cannot win.

The format for instances in this database is a sequence of 37 attribute values.
Each instance is a board-descriptions for this chess endgame. The first
36 attributes describe the board. The last (37th) attribute is the
classification: "win" or "nowin". There are 0 missing values.
A typical board-description is

f,f,f,f,f,f,f,f,f,f,f,f,l,f,n,f,f,t,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,t,t,n,won

The names of the features do not appear in the board-descriptions.
Instead, each feature correponds to a particular position in the
feature-value list. For example, the head of this list is the value
for the feature "bkblk". The following is the list of features, in
the order in which their values appear in the feature-value list:

[bkblk,bknwy,bkon8,bkona,bkspr,bkxbq,bkxcr,bkxwp,blxwp,bxqsq,cntxt,dsopp,dwipd,
hdchk,katri,mulch,qxmsq,r2ar8,reskd,reskr,rimmx,rkxwp,rxmsq,simpl,skach,skewr,
skrxp,spcop,stlmt,thrsk,wkcti,wkna8,wknck,wkovl,wkpos,wtoeg]

In the file, there is one instance (board position) per line.


Num Instances: 3196
Num Attributes: 37
Num Continuous: 0 (Int 0 / Real 0)
Num Discrete: 37
Missing values: 0 / 0.0%
====
Target Variable: class (nominal, 2 distinct): ['won', 'nowin']
====
Features:

bkblk (nominal, 2 distinct): ['f', 't']
bknwy (nominal, 2 distinct): ['f', 't']
bkon8 (nominal, 2 distinct): ['f', 't']
bkona (nominal, 2 distinct): ['f', 't']
bkspr (nominal, 2 distinct): ['f', 't']
bkxbq (nominal, 2 distinct): ['f', 't']
bkxcr (nominal, 2 distinct): ['f', 't']
bkxwp (nominal, 2 distinct): ['f', 't']
blxwp (nominal, 2 distinct): ['f', 't']
bxqsq (nominal, 2 distinct): ['f', 't']
cntxt (nominal, 2 distinct): ['f', 't']
dsopp (nominal, 2 distinct): ['f', 't']
dwipd (nominal, 2 distinct): ['l', 'g']
hdchk (nominal, 2 distinct): ['f', 't']
katri (nominal, 3 distinct): ['n', 'w', 'b']
mulch (nominal, 2 distinct): ['f', 't']
qxmsq (nominal, 2 distinct): ['f', 't']
r2ar8 (nominal, 2 distinct): ['t', 'f']
reskd (nominal, 2 distinct): ['f', 't']
reskr (nominal, 2 distinct): ['f', 't']
rimmx (nominal, 2 distinct): ['f', 't']
rkxwp (nominal, 2 distinct): ['f', 't']
rxmsq (nominal, 2 distinct): ['f', 't']
simpl (nominal, 2 distinct): ['f', 't']
skach (nominal, 2 distinct): ['f', 't']
skewr (nominal, 2 distinct): ['t', 'f']
skrxp (nominal, 2 distinct): ['f', 't']
spcop (nominal, 2 distinct): ['f', 't']
stlmt (nominal, 2 distinct): ['f', 't']
thrsk (nominal, 2 distinct): ['f', 't']
wkcti (nominal, 2 distinct): ['f', 't']
wkna8 (nominal, 2 distinct): ['f', 't']
wknck (nominal, 2 distinct): ['f', 't']
wkovl (nominal, 2 distinct): ['t', 'f']
wkpos (nominal, 2 distinct): ['t', 'f']
wtoeg (nominal, 2 distinct): ['n', 't', 'f']
'''

CONTEXT = "Chess Game King Rook vs King Pawn"
TARGET = CuratedTarget(raw_name="class", new_name="Chess Game Outcome", task_type=SupervisedTask.BINARY,
                       label_mapping={'won': 'Win', 'nowin': 'No Win'})
COLS_TO_DROP = []
FEATURES = []