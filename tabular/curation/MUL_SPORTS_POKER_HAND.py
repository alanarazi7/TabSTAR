from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: poker-hand
====
Examples: 1025009
====
URL: https://www.openml.org/search?type=data&id=1567
====
Description: **Author**: Robert Cattral, Franz Oppacher    
**Source**: UCI    
**Please cite**:   

* Abstract: 
Purpose is to predict poker hands

* Source - Creators:   
Robert Cattral (cattral '@' gmail.com)
Franz Oppacher (oppacher '@' scs.carleton.ca) 
Carleton University, Department of Computer Science 
Intelligent Systems Research Unit 
1125 Colonel By Drive, Ottawa, Ontario, Canada, K1S5B6


* Data Set Information:

Each record is an example of a hand consisting of five playing cards drawn from a standard deck of 52. Each card is described using two attributes (suit and rank), for a total of 10 predictive attributes. There is one Class attribute that describes the "Poker Hand". The order of cards is important, which is why there are 480 possible Royal Flush hands as compared to 4 (one for each suit).


* Attribute Information:

1) S1 "Suit of card #1"    
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}    

2) C1 "Rank of card #1"    
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)    

3) S2 "Suit of card #2"    
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}    

4) C2 "Rank of card #2"   
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)   

5) S3 "Suit of card #3"   
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}   

6) C3 "Rank of card #3"   
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)   

7) S4 "Suit of card #4"   
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}   

8) C4 "Rank of card #4"   
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)   

9) S5 "Suit of card #5"   
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}   

10) C5 "Rank of card 5"   
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)   

11) CLASS "Poker Hand"   
Ordinal (0-9)   

0: Nothing in hand; not a recognized poker hand    
1: One pair; one pair of equal ranks within five cards   
2: Two pairs; two pairs of equal ranks within five cards   
3: Three of a kind; three equal ranks within five cards   
4: Straight; five cards, sequentially ranked with no gaps   
5: Flush; five cards with the same suit   
6: Full house; pair + different rank three of a kind   
7: Four of a kind; four equal ranks within five cards   
8: Straight flush; straight + flush   
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush   


* Relevant Papers:

R. Cattral, F. Oppacher, D. Deugo. Evolutionary Data Mining with Automatic Rule Generalization. Recent Advances in Computers, Computing and Communications, pp.296-300, WSEAS Press, 2002. 
Note: This was a slightly different dataset that had more classes, and was considerably more difficult.
====
Target Variable: Class (nominal, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
====
Features:

V1 (numeric, 4 distinct): ['3', '1', '4', '2']
V2 (numeric, 13 distinct): ['1', '11', '6', '4', '12', '13', '2', '8', '5', '10']
V3 (numeric, 4 distinct): ['1', '4', '3', '2']
V4 (numeric, 13 distinct): ['11', '12', '13', '7', '3', '6', '8', '2', '1', '10']
V5 (numeric, 4 distinct): ['3', '4', '1', '2']
V6 (numeric, 13 distinct): ['7', '10', '1', '2', '8', '5', '11', '12', '13', '4']
V7 (numeric, 4 distinct): ['3', '2', '4', '1']
V8 (numeric, 13 distinct): ['7', '2', '9', '11', '12', '13', '4', '10', '3', '1']
V9 (numeric, 4 distinct): ['1', '4', '3', '2']
V10 (numeric, 13 distinct): ['3', '2', '9', '4', '8', '10', '5', '6', '1', '7']
'''

POKER_CARD_TYPE = {'1': 'Hearts', '2': 'Spades', '3': 'Diamonds', '4': 'Clubs'}

CONTEXT = "Poker Hand"
TARGET = CuratedTarget(raw_name="Class", new_name="Poker Hand", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': "Nothing in hand",
                                      '2': "One pair",
                                      '3': "Two pairs",
                                      '4': "Three of a kind",
                                      '5': "Straight",
                                      '6': "Flush",
                                      '7': "Full house",
                                      '8': "Four of a kind",
                                      '9': "Straight flush",
                                      '10': "Royal flush"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="S1 Suit of card #1", value_mapping=POKER_CARD_TYPE),
            CuratedFeature(raw_name="V2", new_name="C1 Rank of card #1"),
            CuratedFeature(raw_name="V3", new_name="S2 Suit of card #2", value_mapping=POKER_CARD_TYPE),
            CuratedFeature(raw_name="V4", new_name="C2 Rank of card #2"),
            CuratedFeature(raw_name="V5", new_name="S3 Suit of card #3", value_mapping=POKER_CARD_TYPE),
            CuratedFeature(raw_name="V6", new_name="C3 Rank of card #3"),
            CuratedFeature(raw_name="V7", new_name="S4 Suit of card #4", value_mapping=POKER_CARD_TYPE),
            CuratedFeature(raw_name="V8", new_name="C4 Rank of card #4"),
            CuratedFeature(raw_name="V9", new_name="S5 Suit of card #5", value_mapping=POKER_CARD_TYPE),
            CuratedFeature(raw_name="V10", new_name="C5 Rank of card 5")]
