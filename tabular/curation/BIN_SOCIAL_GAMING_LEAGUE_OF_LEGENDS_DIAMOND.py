from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: League-of-Legends-Diamond-Games-(First-15-Minutes)
====
Examples: 48651
====
URL: https://www.openml.org/search?type=data&id=43635
====
Description: Context
Inspired by the following dataset , we have a collection of data on the first 15 minutes of about 50000 Diamond ranked League of Legends matches scraped using Riot's API. 
Can you predict their outcomes?
Content
Data
All matches were collected with the following parameters:
Season: 13
Server: NA1
Rank: Diamond
Tier: I,II,III,IV
Acknowledgements
Thank you to Riot Games for allowing access to their API.
Inspiration
When working on the linked dataset above, we see classification accuracy peak around 70. Given that we have 5 times the amount of data, I wanted to explore how this would improve our results.
====
Features:

Unnamed:_0 (numeric, 48651 distinct): ['0', '32437', '32428', '32429', '32430', '32431', '32432', '32433', '32434', '32435']
matchId (numeric, 48632 distinct): ['3500796520.0', '3497155807.0', '3490145577.0', '3480981015.0', '3478915229.0', '3478859768.0', '3478666008.0', '3503247889.0', '3499306030.0', '3510832225.0']
blue_win (numeric, 2 distinct): ['1', '0']
blueGold (numeric, 11986 distinct): ['25616.0', '26726.0', '25475.0', '25191.0', '25701.0', '24989.0', '25498.0', '25160.0', '25309.0', '26495.0']
blueMinionsKilled (numeric, 259 distinct): ['338.0', '335.0', '349.0', '346.0', '336.0', '333.0', '329.0', '339.0', '343.0', '347.0']
blueJungleMinionsKilled (numeric, 137 distinct): ['84', '80', '76', '88', '72', '92', '68', '83', '96', '79']
blueAvgLevel (numeric, 24 distinct): ['9.2', '9.0', '9.4', '9.6', '8.8', '8.6', '9.8', '8.4', '10.0', '8.2']
redGold (numeric, 11960 distinct): ['24552.0', '24383.0', '25100.0', '27465.0', '25966.0', '27438.0', '25538.0', '26061.0', '24855.0', '24576.0']
redMinionsKilled (numeric, 271 distinct): ['333.0', '346.0', '342.0', '334.0', '339.0', '347.0', '341.0', '330.0', '329.0', '336.0']
redJungleMinionsKilled (numeric, 141 distinct): ['84', '88', '80', '76', '92', '72', '96', '68', '100', '83']
redAvgLevel (numeric, 24 distinct): ['9.2', '9.4', '9.0', '9.6', '8.8', '9.8', '8.6', '8.4', '10.0', '8.2']
blueChampKills (numeric, 35 distinct): ['10', '9', '11', '8', '12', '13', '7', '14', '6', '15']
blueHeraldKills (numeric, 5 distinct): ['1', '2', '0', '3', '4']
blueDragonKills (numeric, 1 distinct): ['0']
blueTowersDestroyed (numeric, 12 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
redChampKills (numeric, 37 distinct): ['10', '9', '11', '8', '12', '7', '13', '6', '14', '15']
redHeraldKills (numeric, 5 distinct): ['1', '2', '0', '3', '4']
redDragonKills (numeric, 1 distinct): ['0']
redTowersDestroyed (numeric, 11 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''

CONTEXT = "League of Legends Diamond Games (First 15 Minutes)"
TARGET = CuratedTarget(raw_name="blue_win", new_name="Blue Team Win", task_type=SupervisedTask.BINARY,
                       label_mapping={'1': 'Yes', '0': 'No'})
COLS_TO_DROP = ["Unnamed:_0", "matchId", "blueDragonKills", "redDragonKills"]
FEATURES = []
