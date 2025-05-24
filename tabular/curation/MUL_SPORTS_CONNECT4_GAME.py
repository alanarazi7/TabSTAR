from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: connect-4
====
Examples: 67557
====
URL: https://www.openml.org/search?type=data&id=40668
====
Description: **Author**: John Tromp  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Connect-4) - 1995  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  

**Connect-4**  
This database contains all legal 8-ply positions in the game of connect-4 in which neither player has won yet, and in which the next move is not forced. Attributes represent board positions on a 6x6 board. The outcome class is the game-theoretical value for the first player (2: win, 1: loss, 0: draw).

### Attribute Information  

The board is numbered like:  
6 . . . . . . .  
5 . . . . . . .  
4 . . . . . . .  
3 . . . . . . .  
2 . . . . . . .  
1 . . . . . . .  
a b c d e f g  

The values represent:  
0: Blank  
1: Taken by Player 1  
2: Taken by Player 2
====
Target Variable: class (nominal, 3 distinct): ['2', '1', '0']
====
Features:

a1 (nominal, 3 distinct): ['0', '2', '1']
a2 (nominal, 3 distinct): ['0', '2', '1']
a3 (nominal, 3 distinct): ['0', '1', '2']
a4 (nominal, 3 distinct): ['0', '1', '2']
a5 (nominal, 3 distinct): ['0', '1', '2']
a6 (nominal, 3 distinct): ['0', '1', '2']
b1 (nominal, 3 distinct): ['2', '0', '1']
b2 (nominal, 3 distinct): ['0', '1', '2']
b3 (nominal, 3 distinct): ['0', '1', '2']
b4 (nominal, 3 distinct): ['0', '1', '2']
b5 (nominal, 3 distinct): ['0', '1', '2']
b6 (nominal, 3 distinct): ['0', '1', '2']
c1 (nominal, 3 distinct): ['2', '1', '0']
c2 (nominal, 3 distinct): ['0', '1', '2']
c3 (nominal, 3 distinct): ['0', '1', '2']
c4 (nominal, 3 distinct): ['0', '1', '2']
c5 (nominal, 3 distinct): ['0', '1', '2']
c6 (nominal, 3 distinct): ['0', '1', '2']
d1 (nominal, 3 distinct): ['0', '2', '1']
d2 (nominal, 3 distinct): ['0', '2', '1']
d3 (nominal, 3 distinct): ['0', '1', '2']
d4 (nominal, 3 distinct): ['0', '1', '2']
d5 (nominal, 3 distinct): ['0', '1', '2']
d6 (nominal, 3 distinct): ['0', '1', '2']
e1 (nominal, 3 distinct): ['0', '1', '2']
e2 (nominal, 3 distinct): ['0', '2', '1']
e3 (nominal, 3 distinct): ['0', '2', '1']
e4 (nominal, 3 distinct): ['0', '1', '2']
e5 (nominal, 3 distinct): ['0', '1', '2']
e6 (nominal, 3 distinct): ['0', '1', '2']
f1 (nominal, 3 distinct): ['0', '1', '2']
f2 (nominal, 3 distinct): ['0', '2', '1']
f3 (nominal, 3 distinct): ['0', '2', '1']
f4 (nominal, 3 distinct): ['0', '1', '2']
f5 (nominal, 3 distinct): ['0', '1', '2']
f6 (nominal, 3 distinct): ['0', '1', '2']
g1 (nominal, 3 distinct): ['0', '1', '2']
g2 (nominal, 3 distinct): ['0', '1', '2']
g3 (nominal, 3 distinct): ['0', '1', '2']
g4 (nominal, 3 distinct): ['0', '1', '2']
g5 (nominal, 3 distinct): ['0', '1', '2']
g6 (nominal, 3 distinct): ['0', '1', '2']
'''

POSITION_MAP = {'0': 'Blank', '1': 'Taken by Player 1', '2': 'Taken by Player 2'}
LOCATIONS = [f"{col}{row}" for col in "abcdefg" for row in range(1, 7)]

CONTEXT = "Connect-4 Game Positions"
TARGET = CuratedTarget(raw_name="class", new_name="Game Outcome", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'0': 'Draw', '1': 'First Player Loses', '2': 'First Player Wins'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=loc, new_name=f"Position {loc}", value_mapping=POSITION_MAP) for loc in LOCATIONS]
