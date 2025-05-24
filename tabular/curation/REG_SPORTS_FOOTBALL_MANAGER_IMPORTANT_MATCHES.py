from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ajinkyablaze/football-manager-data/dataset.csv
====
Examples: 159541
====
URL: https://www.kaggle.com/ajinkyablaze/football-manager-data/dataset.csv
====
Description: 
Football Manager Data (150,000+ players)
Includes information from football manager game.

About Dataset
If you play the game, you might understand the columns by themselves, otherwise sometime in this week, ill update the information and provide all details. (Note: The encoding for names is in UTF-8).

If you want to contribute, send me a message.

All Data is collected from the game : Football Manager 2017.

Try to predict the next messi/ronaldo ?

====
Features:

UID (int64, 159541 distinct): ['1000055', '62057667', '62057536', '62057555', '62057569', '62057579', '62057583', '62057650', '62057652', '62057785']
Name (object, 148640 distinct): ['Juninho', 'Gabriel', 'Paulinho', 'Guilherme', 'Lucas', 'João Paulo', 'Bruno', 'Diego', 'Alex', 'Eduardo']
NationID (int64, 213 distinct): ['776', '1649', '1651', '796', '765', '769', '771', '787', '772', '788']
Born (object, 9474 distinct): ['01-01-1997', '01-01-2000', '01-01-1999', '01-01-1995', '03-01-1997', '01-01-1994', '01-01-1998', '09-01-1997', '08-01-1997', '19-01-1996']
Age (int64, 40 distinct): ['16', '19', '20', '18', '21', '22', '23', '17', '24', '25']
IntCaps (int64, 145 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
IntGoals (int64, 60 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
U21Caps (int64, 48 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
U21Goals (int64, 24 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Height (int64, 61 distinct): ['180', '178', '183', '185', '175', '182', '181', '176', '184', '186']
Weight (int64, 61 distinct): ['0', '70', '75', '72', '80', '78', '73', '74', '77', '76']
AerialAbility (int64, 20 distinct): ['1', '3', '2', '4', '13', '12', '11', '10', '14', '9']
CommandOfArea (int64, 20 distinct): ['1', '3', '2', '4', '9', '10', '8', '7', '11', '12']
Communication (int64, 20 distinct): ['3', '2', '1', '4', '8', '7', '9', '10', '6', '11']
Eccentricity (int64, 20 distinct): ['1', '2', '3', '4', '7', '8', '6', '5', '9', '10']
Handling (int64, 19 distinct): ['1', '2', '3', '4', '13', '12', '11', '10', '9', '14']
Kicking (int64, 20 distinct): ['1', '3', '2', '4', '10', '11', '8', '9', '7', '12']
OneOnOnes (int64, 19 distinct): ['1', '3', '2', '4', '8', '9', '7', '10', '11', '6']
Reflexes (int64, 20 distinct): ['1', '2', '3', '4', '13', '12', '14', '11', '10', '9']
RushingOut (int64, 20 distinct): ['1', '3', '2', '4', '8', '7', '9', '6', '10', '5']
TendencyToPunch (int64, 20 distinct): ['1', '3', '2', '4', '8', '7', '9', '6', '10', '5']
Throwing (int64, 20 distinct): ['3', '1', '2', '4', '7', '8', '6', '9', '5', '10']
Corners (int64, 20 distinct): ['5', '6', '4', '3', '7', '1', '2', '8', '9', '10']
Crossing (int64, 19 distinct): ['1', '3', '2', '6', '7', '5', '8', '4', '9', '10']
Dribbling (int64, 20 distinct): ['1', '3', '8', '2', '10', '9', '11', '7', '12', '6']
Finishing (int64, 20 distinct): ['1', '3', '6', '7', '2', '5', '8', '4', '9', '10']
FirstTouch (int64, 20 distinct): ['10', '11', '9', '12', '8', '13', '7', '6', '5', '1']
Freekicks (int64, 19 distinct): ['6', '1', '5', '7', '4', '8', '3', '9', '2', '10']
Heading (int64, 20 distinct): ['1', '7', '6', '8', '5', '9', '10', '12', '13', '11']
LongShots (int64, 19 distinct): ['7', '6', '1', '3', '8', '5', '2', '4', '9', '10']
Longthrows (int64, 20 distinct): ['1', '3', '2', '5', '4', '6', '7', '8', '9', '10']
Marking (int64, 19 distinct): ['6', '5', '3', '7', '8', '4', '9', '1', '2', '10']
Passing (int64, 20 distinct): ['8', '10', '9', '7', '11', '6', '12', '1', '5', '4']
PenaltyTaking (int64, 20 distinct): ['1', '3', '2', '4', '5', '6', '7', '8', '10', '9']
Tackling (int64, 20 distinct): ['1', '3', '6', '12', '11', '7', '8', '5', '10', '2']
Technique (int64, 20 distinct): ['10', '9', '11', '8', '12', '7', '6', '13', '5', '1']
Aggression (int64, 20 distinct): ['8', '9', '7', '6', '10', '12', '5', '11', '13', '14']
Anticipation (int64, 20 distinct): ['10', '9', '8', '11', '7', '12', '6', '5', '4', '13']
Bravery (int64, 20 distinct): ['8', '7', '9', '6', '10', '12', '5', '11', '13', '4']
Composure (int64, 20 distinct): ['8', '7', '9', '10', '6', '5', '11', '4', '12', '3']
Concentration (int64, 19 distinct): ['8', '7', '9', '10', '6', '11', '5', '12', '4', '3']
Vision (int64, 20 distinct): ['8', '7', '6', '9', '10', '5', '11', '4', '12', '3']
Decisions (int64, 20 distinct): ['12', '11', '10', '13', '9', '8', '14', '7', '6', '5']
Determination (int64, 20 distinct): ['8', '9', '7', '4', '6', '3', '5', '12', '10', '2']
Flair (int64, 20 distinct): ['8', '9', '7', '1', '10', '6', '3', '11', '12', '2']
Leadership (int64, 20 distinct): ['10', '9', '11', '8', '7', '12', '6', '5', '13', '4']
OffTheBall (int64, 20 distinct): ['7', '8', '6', '9', '10', '3', '5', '11', '12', '4']
Positioning (int64, 20 distinct): ['7', '8', '6', '9', '10', '11', '5', '12', '4', '13']
Teamwork (int64, 20 distinct): ['7', '8', '6', '9', '5', '10', '12', '11', '13', '4']
Workrate (int64, 20 distinct): ['11', '10', '12', '8', '9', '7', '6', '13', '5', '14']
Acceleration (int64, 20 distinct): ['13', '12', '11', '14', '10', '9', '8', '7', '15', '6']
Agility (int64, 20 distinct): ['12', '11', '13', '10', '9', '14', '8', '7', '6', '15']
Balance (int64, 20 distinct): ['7', '8', '6', '9', '10', '5', '11', '12', '4', '13']
Jumping (int64, 20 distinct): ['8', '9', '7', '10', '11', '6', '12', '5', '13', '4']
LeftFoot (int64, 20 distinct): ['20', '8', '7', '9', '10', '5', '6', '11', '12', '1']
NaturalFitness (int64, 20 distinct): ['12', '13', '10', '11', '14', '9', '8', '15', '7', '16']
Pace (int64, 20 distinct): ['12', '13', '11', '10', '14', '9', '8', '7', '15', '6']
RightFoot (int64, 20 distinct): ['20', '8', '5', '10', '7', '9', '6', '11', '12', '4']
Stamina (int64, 20 distinct): ['12', '11', '10', '8', '7', '13', '9', '6', '5', '14']
Strength (int64, 20 distinct): ['8', '7', '9', '6', '10', '11', '5', '12', '13', '4']
Consistency (int64, 20 distinct): ['9', '8', '7', '10', '6', '12', '11', '13', '5', '14']
Dirtiness (int64, 20 distinct): ['8', '7', '9', '6', '5', '10', '4', '11', '12', '3']
ImportantMatches (int64, 20 distinct): ['8', '9', '7', '10', '6', '5', '12', '11', '4', '3']
InjuryProness (int64, 20 distinct): ['8', '7', '9', '6', '5', '10', '4', '3', '11', '12']
Versatility (int64, 20 distinct): ['9', '8', '7', '10', '6', '11', '12', '5', '13', '14']
Adaptability (int64, 20 distinct): ['10', '12', '11', '9', '8', '13', '14', '7', '15', '16']
Ambition (int64, 20 distinct): ['11', '12', '10', '13', '9', '14', '8', '15', '7', '16']
Loyalty (int64, 20 distinct): ['11', '12', '10', '13', '9', '14', '8', '15', '7', '16']
Pressure (int64, 20 distinct): ['11', '10', '12', '13', '9', '8', '14', '7', '6', '15']
Professional (int64, 20 distinct): ['11', '12', '10', '13', '9', '14', '8', '15', '16', '7']
Sportsmanship (int64, 20 distinct): ['11', '10', '12', '9', '13', '8', '14', '7', '6', '5']
Temperament (int64, 20 distinct): ['12', '10', '13', '11', '14', '15', '20', '16', '17', '18']
Controversy (int64, 20 distinct): ['6', '8', '7', '5', '4', '3', '2', '1', '9', '10']
PositionsDesc (object, 286 distinct): ['S ', 'D C', 'GK ', 'M C', 'DM/M C', 'D R', 'D L', 'D RC', 'AM C', 'M/AM C']
Goalkeeper (int64, 13 distinct): ['1', '20', '5', '3', '10', '4', '2', '8', '14', '7']
Sweeper (int64, 20 distinct): ['1', '10', '12', '13', '15', '20', '11', '14', '5', '7']
Striker (int64, 20 distinct): ['1', '20', '10', '15', '14', '12', '13', '17', '18', '16']
AttackingMidCentral (int64, 20 distinct): ['1', '20', '10', '15', '14', '13', '12', '17', '18', '16']
AttackingMidLeft (int64, 20 distinct): ['1', '20', '15', '10', '17', '14', '12', '13', '18', '16']
AttackingMidRight (int64, 20 distinct): ['1', '20', '15', '10', '17', '14', '18', '12', '13', '16']
DefenderCentral (int64, 20 distinct): ['1', '20', '10', '15', '14', '18', '17', '12', '13', '16']
DefenderLeft (int64, 20 distinct): ['1', '20', '10', '15', '14', '17', '12', '18', '13', '16']
DefenderRight (int64, 20 distinct): ['1', '20', '15', '10', '17', '14', '12', '18', '13', '16']
DefensiveMidfielder (int64, 20 distinct): ['1', '20', '10', '15', '18', '14', '17', '13', '12', '16']
MidfielderCentral (int64, 20 distinct): ['1', '20', '15', '10', '18', '17', '14', '13', '12', '16']
MidfielderLeft (int64, 20 distinct): ['1', '20', '10', '15', '14', '12', '13', '17', '18', '16']
MidfielderRight (int64, 20 distinct): ['1', '20', '10', '15', '14', '12', '13', '17', '18', '16']
WingBackLeft (int64, 20 distinct): ['1', '10', '15', '18', '14', '20', '17', '13', '7', '12']
WingBackRight (int64, 20 distinct): ['1', '10', '15', '14', '18', '17', '12', '13', '7', '20']
'''

CONTEXT = "Football Manager Stats: Important Matches"
TARGET = CuratedTarget(raw_name="ImportantMatches", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["UID"]
FEATURES = []

DESCRIPTION = '''
Football Manager Data (150,000+ players)
Includes information from football manager game.

About Dataset
If you play the game, you might understand the columns by themselves, otherwise sometime in this week, ill update the information and provide all details. (Note: The encoding for names is in UTF-8).

If you want to contribute, send me a message.

All Data is collected from the game : Football Manager 2017.

Try to predict the next messi/ronaldo ?
'''