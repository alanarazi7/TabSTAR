from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Football_players_Fifa_stats
====
Examples: 183978
====
URL: https://www.openml.org/search?type=data&id=42194
====
Description: The dataset contains all the statistics for each player from 2008 to 2016.
====
Features:

id (numeric, 183978 distinct): ['1', '122666', '122646', '122647', '122648', '122649', '122650', '122651', '122652', '122653']
player_fifa_api_id (numeric, 11062 distinct): ['184431', '178393', '193061', '155887', '173210', '184267', '168650', '41635', '183284', '179605']
player_api_id (numeric, 11060 distinct): ['41269', '210278', '42116', '26472', '179795', '41890', '25772', '30453', '47394', '163838']
date (string, 197 distinct): ['2007-02-22 00:00:00', '2013-09-20 00:00:00', '2011-08-30 00:00:00', '2015-09-21 00:00:00', '2012-08-31 00:00:00', '2014-09-18 00:00:00', '2013-02-15 00:00:00', '2010-08-30 00:00:00', '2012-02-22 00:00:00', '2009-08-30 00:00:00']
overall_rating (numeric, 62 distinct): ['68.0', '69.0', '67.0', '66.0', '70.0', '72.0', '71.0', '65.0', '73.0', '64.0']
potential (numeric, 57 distinct): ['75.0', '74.0', '76.0', '72.0', '73.0', '70.0', '78.0', '77.0', '71.0', '69.0']
preferred_foot (string, 3 distinct): ['right', 'left']
attacking_work_rate (string, 9 distinct): ['medium', 'high', 'low', 'None', 'norm', 'y', 'le', 'stoc']
defensive_work_rate (string, 20 distinct): ['medium', 'high', 'low', '_0', 'o', '1', 'ormal', '2', '3', '5']
crossing (numeric, 96 distinct): ['68.0', '62.0', '67.0', '60.0', '64.0', '25.0', '63.0', '65.0', '66.0', '61.0']
finishing (numeric, 98 distinct): ['25.0', '64.0', '66.0', '65.0', '62.0', '60.0', '68.0', '63.0', '67.0', '58.0']
heading_accuracy (numeric, 97 distinct): ['68.0', '60.0', '64.0', '58.0', '62.0', '65.0', '67.0', '59.0', '66.0', '63.0']
short_passing (numeric, 96 distinct): ['65.0', '64.0', '68.0', '66.0', '67.0', '62.0', '72.0', '74.0', '63.0', '69.0']
volleys (numeric, 94 distinct): ['25.0', '59.0', '58.0', '60.0', '64.0', '56.0', '63.0', '68.0', '57.0', '66.0']
dribbling (numeric, 98 distinct): ['68.0', '66.0', '64.0', '72.0', '67.0', '62.0', '65.0', '73.0', '69.0', '74.0']
curve (numeric, 93 distinct): ['25.0', '68.0', '60.0', '63.0', '66.0', '64.0', '58.0', '59.0', '67.0', '56.0']
free_kick_accuracy (numeric, 98 distinct): ['25.0', '60.0', '58.0', '54.0', '42.0', '52.0', '41.0', '46.0', '56.0', '48.0']
long_passing (numeric, 96 distinct): ['64.0', '65.0', '62.0', '58.0', '60.0', '66.0', '68.0', '61.0', '67.0', '63.0']
ball_control (numeric, 94 distinct): ['68.0', '74.0', '73.0', '66.0', '72.0', '67.0', '70.0', '64.0', '65.0', '71.0']
acceleration (numeric, 87 distinct): ['68.0', '69.0', '74.0', '67.0', '76.0', '75.0', '78.0', '77.0', '66.0', '72.0']
sprint_speed (numeric, 86 distinct): ['68.0', '69.0', '76.0', '74.0', '75.0', '67.0', '78.0', '72.0', '73.0', '66.0']
agility (numeric, 82 distinct): ['72.0', '74.0', '73.0', '68.0', '70.0', '75.0', '65.0', '71.0', '76.0', '64.0']
reactions (numeric, 79 distinct): ['68.0', '66.0', '70.0', '67.0', '64.0', '69.0', '65.0', '72.0', '71.0', '74.0']
balance (numeric, 82 distinct): ['70.0', '72.0', '71.0', '74.0', '68.0', '73.0', '65.0', '75.0', '62.0', '60.0']
shot_power (numeric, 97 distinct): ['68.0', '72.0', '74.0', '70.0', '71.0', '65.0', '67.0', '75.0', '69.0', '73.0']
jumping (numeric, 80 distinct): ['72.0', '70.0', '71.0', '73.0', '74.0', '64.0', '66.0', '68.0', '67.0', '62.0']
stamina (numeric, 85 distinct): ['68.0', '72.0', '70.0', '74.0', '69.0', '75.0', '73.0', '71.0', '67.0', '65.0']
strength (numeric, 83 distinct): ['68.0', '74.0', '72.0', '70.0', '75.0', '69.0', '67.0', '76.0', '73.0', '78.0']
long_shots (numeric, 97 distinct): ['25.0', '64.0', '68.0', '60.0', '66.0', '63.0', '65.0', '62.0', '67.0', '58.0']
aggression (numeric, 92 distinct): ['68.0', '74.0', '72.0', '75.0', '70.0', '71.0', '73.0', '67.0', '69.0', '66.0']
interceptions (numeric, 97 distinct): ['25.0', '64.0', '68.0', '65.0', '72.0', '69.0', '66.0', '67.0', '63.0', '70.0']
positioning (numeric, 96 distinct): ['25.0', '68.0', '64.0', '66.0', '67.0', '65.0', '70.0', '62.0', '72.0', '69.0']
vision (numeric, 98 distinct): ['68.0', '64.0', '65.0', '58.0', '59.0', '62.0', '66.0', '70.0', '67.0', '60.0']
penalties (numeric, 95 distinct): ['58.0', '64.0', '60.0', '62.0', '68.0', '54.0', '59.0', '56.0', '66.0', '65.0']
marking (numeric, 96 distinct): ['25.0', '68.0', '64.0', '22.0', '65.0', '21.0', '67.0', '66.0', '62.0', '63.0']
standing_tackle (numeric, 96 distinct): ['25.0', '68.0', '66.0', '65.0', '67.0', '72.0', '71.0', '69.0', '64.0', '74.0']
sliding_tackle (numeric, 95 distinct): ['25.0', '68.0', '65.0', '64.0', '67.0', '66.0', '63.0', '62.0', '72.0', '70.0']
gk_diving (numeric, 94 distinct): ['8.0', '9.0', '7.0', '6.0', '11.0', '10.0', '12.0', '13.0', '14.0', '5.0']
gk_handling (numeric, 91 distinct): ['7.0', '11.0', '14.0', '8.0', '9.0', '10.0', '12.0', '13.0', '6.0', '15.0']
gk_kicking (numeric, 98 distinct): ['7.0', '12.0', '10.0', '9.0', '8.0', '11.0', '13.0', '6.0', '14.0', '15.0']
gk_positioning (numeric, 95 distinct): ['9.0', '7.0', '8.0', '14.0', '10.0', '13.0', '11.0', '6.0', '12.0', '15.0']
gk_reflexes (numeric, 93 distinct): ['8.0', '10.0', '14.0', '7.0', '9.0', '13.0', '12.0', '6.0', '11.0', '15.0']
'''

CONTEXT = "FIFA Players Stats 2008-2016"
TARGET = CuratedTarget(raw_name="overall_rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id", "player_fifa_api_id", "player_api_id", "date"]
FEATURES = []