from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: tic-tac-toe
====
Examples: 958
====
URL: https://www.openml.org/search?type=data&id=50
====
Description: **Author**: David W. Aha    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame) - 1991   
**Please cite**: [UCI](http://archive.ics.uci.edu/ml/citation_policy.html)

**Tic-Tac-Toe Endgame database**  
This database encodes the complete set of possible board configurations at the end of tic-tac-toe games, where "x" is assumed to have played first.  The target concept is "win for x" (i.e., true when "x" has one of 8 possible ways to create a "three-in-a-row").  

### Attribute Information  

     (x=player x has taken, o=player o has taken, b=blank)
     1. top-left-square: {x,o,b}
     2. top-middle-square: {x,o,b}
     3. top-right-square: {x,o,b}
     4. middle-left-square: {x,o,b}
     5. middle-middle-square: {x,o,b}
     6. middle-right-square: {x,o,b}
     7. bottom-left-square: {x,o,b}
     8. bottom-middle-square: {x,o,b}
     9. bottom-right-square: {x,o,b}
    10. Class: {positive,negative}
====
Target Variable: Class (nominal, 2 distinct): ['positive', 'negative']
====
Features:

top-left-square (nominal, 3 distinct): ['x', 'o', 'b']
top-middle-square (nominal, 3 distinct): ['x', 'o', 'b']
top-right-square (nominal, 3 distinct): ['x', 'o', 'b']
middle-left-square (nominal, 3 distinct): ['x', 'o', 'b']
middle-middle-square (nominal, 3 distinct): ['x', 'o', 'b']
middle-right-square (nominal, 3 distinct): ['x', 'o', 'b']
bottom-left-square (nominal, 3 distinct): ['x', 'o', 'b']
bottom-middle-square (nominal, 3 distinct): ['x', 'o', 'b']
bottom-right-square (nominal, 3 distinct): ['x', 'o', 'b']
'''

X_O_DICT = {'x': 'Player X Taken', 'o': 'Player O Taken', 'b': 'Blank'}

CONTEXT = "Tic Tac Toe Game between players X and O"
TARGET = CuratedTarget(raw_name="Class", new_name="Result", task_type=SupervisedTask.BINARY,
                       label_mapping={"positive": "PlayerX won", "negative": "Player O won"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="top-left-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="top-middle-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="top-right-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="middle-left-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="middle-middle-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="middle-right-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="bottom-left-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="bottom-middle-square", value_mapping=X_O_DICT),
            CuratedFeature(raw_name="bottom-right-square", value_mapping=X_O_DICT)]
