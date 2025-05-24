from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: jungle_chess_2pcs_raw_endgame_complete
====
Examples: 44819
====
URL: https://www.openml.org/search?type=data&id=41027
====
Description: ### Description ###

This dataset is part of a collection datasets based on the game "Jungle Chess" (a.k.a. Dou Shou Qi). For a description of the rules, please refer to the paper (link attached). The paper also contains a description of various constructed features. As the tablebases are a disjoint set of several tablebases based on which (two) pieces are on the board, we have uploaded all tablebases that have explicit different content:

* Rat vs Rat
* Rat vs Panther
* Rat vs. Lion
* Rat vs. Elephant
* Panther vs. Lion
* Panther vs. Elephant
* Tiger vs. Lion
* Lion vs. Lion
* Lion vs. Elephant
* Elephant vs. Elephant
* Complete (Combination of the above)
* RAW Complete (Combination of the above, containing for both pieces just the rank, file and strength information). This dataset contains a similar classification problem as, e.g., the King and Rook vs. King problem and is suitable for classification tasks. 

(Note that this dataset is one of the above mentioned datasets). Additionally, note that several subproblems are very similar. Having seen a given positions from one of the tablebases arguably gives a lot of information about the outcome of the same position in the other tablebases. 

### Please cite ###
J. N. van Rijn and J. K. Vis, Endgame Analysis of Dou Shou Qi. ICGA Journal 37:2, 120--124, 2014. ArXiv link: https://arxiv.org/abs/1604.07312
====
Target Variable: class (nominal, 3 distinct): ['w', 'b', 'd']
====
Features:

white_piece0_strength (numeric, 5 distinct): ['0', '6', '7', '4', '5']
white_piece0_file (numeric, 7 distinct): ['6', '0', '3', '1', '2', '4', '5']
white_piece0_rank (numeric, 9 distinct): ['7', '6', '2', '1', '8', '0', '5', '4', '3']
black_piece0_strength (numeric, 5 distinct): ['0', '6', '7', '4', '5']
black_piece0_file (numeric, 7 distinct): ['0', '6', '3', '1', '2', '4', '5']
black_piece0_rank (numeric, 9 distinct): ['7', '6', '2', '1', '8', '0', '5', '4', '3']
'''

CONTEXT = "Jungle Chess 2pcs Game Endgame"
TARGET = CuratedTarget(raw_name="class", new_name="Result", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'w': 'White Wins', 'b': 'Black Wins', 'd': 'Draw'})
COLS_TO_DROP = []
FEATURES = []