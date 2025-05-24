from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: NBA-PLAYERS--2016-2019
====
Examples: 1408
====
URL: https://www.openml.org/search?type=data&id=43653
====
Description: Context
This Dataset was created for an University project in Milan. The goal of the project was to create a Robust Model to predict the All Star Game score for each player. 
The score is calculated by dividing the players by conference and position held in the field (external or internal),
the athletes are ranked in descending order depending on the number of votes taken for each category
voter, so as to obtain three different rankings. From these for each player the rank is calculated (1st
Position involves Rank = 1, 2nd Position Rank = 2 etc. . . ). With the values obtained then an average is calculated
weighed, with weight 0.50 to that of the votes of the fans and 0.25 to the two remaining.
Doing this work we have merged some datasets from kaggle (https://www.kaggle.com/noahgift/social-power-nba),  basketball-reference.com  and hoopshype.com.
Obviously for our work we didn't use all of the variables and for a problem of indipendent observations we took only the last seasons' observation for each player.
Other analysis could be performed using salary as a target but also a cluster analysis for players or a PCA.
We wouldn't be here without the help of others. Thank you Riccardo,Alfredo and Daniel.
Some variables:
POS1= Main position (some players have a second position called POS2)
G= Games played
GS= Games started
MP= Minutes played
FG =Field Goals Per Game
FGA=Field Goal Attempts Per Game
FG.= Field Goal Percentage
X3P= 3-Point Field Goals Per Game
X3PA= 3-Point Field Goal Attempts Per Game
X3P.= FG on 3-Pt FGAs.
X2P =2-Point Field Goals Per Game
X2PA =2-Point Field Goal Attempts Per Game
X2P.= FG on 2-Pt FGAs.
eFG. = Effective Field Goal Percentage
FT=Free Throws Per Game
FTA = Free Throw Attempts Per Game
FT.= Free Throw Percentage
ORB = Offensive Rebounds Per Game
DRB = Defensive Rebounds Per Game
TRB = Total Rebounds Per Game
AST = Assists Per Game
STL= Steals Per Game
BLK = Blocks Per Game
TOV = Turnovers Per Game
PF = Personal Fouls Per Game
PTS =Points Per Game
MEAN_VIEWS= Daily views on wikipedia
PLAY= If the player played in the all star game
====
Features:

Rk (numeric, 539 distinct): ['170', '39', '358', '468', '357', '410', '79', '311', '190', '429']
Player.x (string, 660 distinct): ['Paul George', 'Lance Thomas', 'Malcolm Brogdon', 'Malachi Richardson', 'Luol Deng', 'Zaza Pachulia', 'Luc Mbah a Moute', 'Lou Williams', 'LeBron James', 'Langston Galloway']
Player_ID (string, 660 distinct): ['georgpa01', 'thomala01', 'brogdma01', 'richama01', 'denglu01', 'pachuza01', 'mbahalu01', 'willilo02', 'jamesle01', 'gallola01']
Pos1 (string, 5 distinct): ['SG', 'PF', 'PG', 'C', 'SF']
Pos2 (string, 5 distinct): ['SG', 'SF', 'C', 'PF']
Age (numeric, 24 distinct): ['23', '24', '25', '22', '26', '28', '27', '21', '29', '30']
Tm (string, 30 distinct): ['MIL', 'PHI', 'IND', 'DET', 'LAC', 'UTA', 'GSW', 'DEN', 'MEM', 'CHI']
G (numeric, 82 distinct): ['82', '81', '74', '80', '73', '77', '75', '79', '76', '78']
GS (numeric, 83 distinct): ['0', '1', '2', '3', '4', '6', '81', '5', '13', '80']
MP (numeric, 344 distinct): ['20.0', '34.0', '21.2', '23.5', '34.2', '15.5', '16.5', '25.9', '12.0', '18.4']
FG (numeric, 104 distinct): ['2.0', '2.5', '3.0', '1.0', '1.5', '2.7', '2.1', '2.8', '2.2', '0.7']
FGA (numeric, 200 distinct): ['3.0', '2.0', '4.4', '5.4', '3.5', '5.9', '5.1', '4.1', '5.5', '6.4']
FG. (numeric, 332 distinct): ['0.5', '0.444', '0.452', '0.434', '0.4', '0.423', '0.412', '0.442', '0.402', '0.429']
X3P (numeric, 41 distinct): ['0.0', '0.5', '0.3', '0.1', '0.8', '0.4', '0.7', '0.6', '0.2', '0.9']
X3PA (numeric, 90 distinct): ['0.0', '0.1', '0.8', '1.2', '1.7', '0.2', '2.6', '1.4', '2.7', '0.5']
X3P. (numeric, 261 distinct): ['0.0', '0.333', '0.2', '0.367', '0.25', '0.5', '0.348', '0.4', '0.324', '0.371']
X2P (numeric, 85 distinct): ['1.2', '1.0', '0.9', '0.5', '1.1', '0.7', '1.5', '0.8', '1.7', '2.1']
X2PA (numeric, 154 distinct): ['2.1', '2.8', '1.8', '3.4', '1.5', '2.0', '3.1', '3.2', '2.7', '4.6']
X2P. (numeric, 338 distinct): ['0.5', '0.333', '0.483', '0.477', '0.494', '0.556', '0.481', '0.528', '0.492', '0.514']
eFG. (numeric, 312 distinct): ['0.5', '0.523', '0.518', '0.516', '0.492', '0.506', '0.507', '0.481', '0.54', '0.504']
FT (numeric, 76 distinct): ['0.4', '0.7', '0.5', '0.8', '0.6', '0.9', '0.3', '0.0', '1.2', '1.0']
FTA (numeric, 89 distinct): ['0.8', '1.0', '0.5', '0.9', '0.7', '1.3', '1.2', '0.6', '0.4', '0.0']
FT. (numeric, 371 distinct): ['0.667', '0.5', '1.0', '0.75', '0.8', '0.6', '0.818', '0.833', '0.806', '0.772']
ORB (numeric, 46 distinct): ['0.3', '0.4', '0.5', '0.2', '0.6', '0.7', '0.8', '0.1', '0.9', '1.0']
DRB (numeric, 96 distinct): ['1.8', '2.1', '2.4', '2.7', '2.3', '1.3', '2.0', '1.2', '3.2', '2.2']
TRB (numeric, 122 distinct): ['2.5', '1.8', '2.3', '1.9', '3.1', '3.8', '2.9', '4.0', '2.6', '2.8']
AST (numeric, 91 distinct): ['0.5', '1.2', '1.0', '0.6', '0.9', '0.7', '1.1', '0.4', '0.8', '1.3']
STL (numeric, 24 distinct): ['0.5', '0.3', '0.4', '0.6', '0.7', '0.2', '0.8', '0.9', '1.0', '0.1']
BLK (numeric, 28 distinct): ['0.1', '0.2', '0.3', '0.4', '0.0', '0.5', '0.6', '0.7', '0.8', '0.9']
TOV (numeric, 47 distinct): ['0.8', '0.5', '0.6', '0.9', '0.7', '0.4', '1.0', '0.3', '1.3', '1.1']
PF (numeric, 41 distinct): ['1.6', '2.0', '1.9', '1.8', '1.7', '2.2', '1.5', '1.4', '2.1', '2.3']
PTS (numeric, 248 distinct): ['4.0', '6.7', '5.9', '5.0', '2.3', '6.3', '6.9', '4.4', '2.5', '7.1']
Salary (numeric, 833 distinct): ['77250.0', '1312611.0', '543471.0', '1378242.0', '815615.0', '1471382.0', '838464.0', '1544951.0', '2393887.0', '5000000.0']
mean_views (numeric, 1085 distinct): ['209.6038', '583.2186', '1233.724', '11.0989', '5.0685', '444.0956', '225.0683', '276.0328', '377.9481', '1264.959']
Season (string, 3 distinct): ['2017-18', '2018-19', '2016-17']
Conference (string, 2 distinct): ['West', 'Est']
Role (string, 2 distinct): ['Front', 'Back']
Fvot (numeric, 1357 distinct): ['136', '1268', '3744', '4452', '369', '2012', '2695', '1732', '1283', '10621']
FRank (numeric, 145 distinct): ['68', '70', '60', '47', '23', '25', '2', '6', '59', '44']
Pvot (numeric, 86 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '10.0']
PRank (numeric, 52 distinct): ['74.0', '52.0', '60.0', '53.0', '36.0', '61.0', '84.0', '26.0', '88.0', '40.0']
Mvot (numeric, 39 distinct): ['0.0', '1.0', '2.0', '99.0', '4.0', '11.0', '94.0', '66.0', '34.0', '5.0']
MRank (numeric, 10 distinct): ['8.0', '7.0', '6.0', '9.0', '4.0', '1.0', '3.0', '2.0', '5.0']
Score (numeric, 572 distinct): ['53.8', '43.0', '62.0', '43.5', '45.0', '89.8', '42.5', '42.0', '31.0', '52.8']
Play (string, 2 distinct): ['No', 'Yes']
'''

NBA_POSITIONS = {"SG": "Shooting Guard", 
                 "PF": "Power Forward", 
                 "PG": "Point Guard", 
                 "C": "Center", 
                 "SF": "Small Forward"}

POS2 = {p: v for p, v in NBA_POSITIONS.items() if p != "PG"}


CONTEXT = "NBA Players All Star Game Performance"
TARGET = CuratedTarget(raw_name="Rk", new_name="NBA All-Star Rank", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Player_ID"]
FEATURES = [CuratedFeature(raw_name="Player.x", new_name="NBA Player Name"),
            CuratedFeature(raw_name="Pos1", new_name="Primary Position", value_mapping=NBA_POSITIONS),
            CuratedFeature(raw_name="Pos2", new_name="Secondary Position", value_mapping=POS2),
            CuratedFeature(raw_name="Age"),
            CuratedFeature(raw_name="Tm", new_name="NBA Team"),
            CuratedFeature(raw_name="G", new_name="Games Played"),
            CuratedFeature(raw_name="GS", new_name="Games Started"),
            CuratedFeature(raw_name="MP", new_name="Minutes Played"),
            CuratedFeature(raw_name="FG", new_name="Field Goals Per Game"),
            CuratedFeature(raw_name="FGA", new_name="Field Goal Attempts Per Game"),
            CuratedFeature(raw_name="FG.", new_name="Field Goal Percentage"),
            CuratedFeature(raw_name="X3P", new_name="3-Point Field Goals Per Game"),
            CuratedFeature(raw_name="X3PA", new_name=" 3-Point Field Goal Attempts Per Game"),
            CuratedFeature(raw_name="X3P.", new_name="FG on 3-Pt FGAs"),
            CuratedFeature(raw_name="X2P", new_name="2-Point Field Goals Per Game"),
            CuratedFeature(raw_name="X2PA", new_name="2-Point Field Goal Attempts Per Game"),
            CuratedFeature(raw_name="X2P.", new_name="FG on 2-Pt FGAs"),
            CuratedFeature(raw_name="eFG.", new_name="Effective Field Goal Percentage"),
            CuratedFeature(raw_name="FT", new_name="Free Throws Per Game"),
            CuratedFeature(raw_name="FTA", new_name="Free Throw Attempts Per Game"),
            CuratedFeature(raw_name="FT.", new_name="Free Throw Percentage"),
            CuratedFeature(raw_name="ORB", new_name="Offensive Rebounds Per Game"),
            CuratedFeature(raw_name="DRB", new_name="Defensive Rebounds Per Game"),
            CuratedFeature(raw_name="TRB", new_name="Total Rebounds Per Game"),
            CuratedFeature(raw_name="AST", new_name="Assists Per Game"),
            CuratedFeature(raw_name="STL", new_name="Steals Per Game"),
            CuratedFeature(raw_name="BLK", new_name="Blocks Per Game"),
            CuratedFeature(raw_name="TOV", new_name="Turnovers Per Game"),
            CuratedFeature(raw_name="PF", new_name="Personal Fouls Per Game"),
            CuratedFeature(raw_name="PTS", new_name="Points Per Game"),
            CuratedFeature(raw_name="Salary", new_name="NBA Salary"),
            CuratedFeature(raw_name="mean_views", new_name="NBA Mean Views"),
            CuratedFeature(raw_name="Conference", new_name="NBA Conference",
                           value_mapping={"West": "Western Conference", "Est": "Eastern Conference"}),
            CuratedFeature(raw_name="Role", new_name="NBA Player Role"),
            CuratedFeature(raw_name="Fvot", new_name="NBA Fan Votes"),
            CuratedFeature(raw_name="FRank", new_name="NBA Fan Rank"),
            CuratedFeature(raw_name="Pvot", new_name="NBA Player Votes"),
            CuratedFeature(raw_name="PRank", new_name="NBA Player Rank"),
            CuratedFeature(raw_name="Mvot", new_name="NBA Media Votes"),
            CuratedFeature(raw_name="MRank", new_name="NBA Media Rank"),
            CuratedFeature(raw_name="Play", new_name="Played in NBA All Star", value_mapping={"No": "No", "Yes": "Yes"})
            ]
