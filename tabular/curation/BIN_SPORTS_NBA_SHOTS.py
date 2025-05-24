from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: nba-shot-logs
====
Examples: 128069
====
URL: https://www.openml.org/search?type=data&id=42806
====
Description: https://www.kaggle.com/dansbecker/nba-shot-logs

(from Kaggle)

Data on shots taken during the 2014-2015 season, who took the shot, where on the floor was the shot taken from, who was the nearest defender, how far away was the nearest defender, time on the shot clock, and much more. The column titles are generally self-explanatory.

Useful for evaluating who the best shooter is, who the best defender is, the hot-hand hypothesis, etc.

Scraped from NBA's REST API.


====
Features:

GAME_ID (numeric, 904 distinct): ['21400248.0', '21400375.0', '21400440.0', '21400266.0', '21400695.0', '21400560.0', '21400679.0', '21400390.0', '21400071.0', '21400531.0']
MATCHUP (nominal, 1808 distinct): ['FEB 07, 2015 - DAL vs. POR', 'JAN 29, 2015 - CHI @ LAL', 'NOV 30, 2014 - TOR @ LAL', 'OCT 29, 2014 - CHA vs. MIL', 'DEC 03, 2014 - SAS @ BKN', 'JAN 21, 2015 - WAS vs. OKC', 'DEC 17, 2014 - DEN vs. HOU', 'DEC 04, 2014 - GSW vs. NOP', 'FEB 06, 2015 - GSW @ ATL', 'DEC 08, 2014 - BOS @ WAS']
LOCATION (nominal, 2 distinct): ['A', 'H']
W (nominal, 2 distinct): ['W', 'L']
FINAL_MARGIN (numeric, 88 distinct): ['-5.0', '7.0', '-7.0', '5.0', '-8.0', '8.0', '-2.0', '-6.0', '4.0', '2.0']
SHOT_NUMBER (numeric, 38 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
PERIOD (numeric, 7 distinct): ['1', '3', '2', '4', '5', '6', '7']
GAME_CLOCK (nominal, 719 distinct): ['0:01', '0:02', '0:00', '0:03', '0:04', '0:05', '11:46', '0:32', '0:34', '11:45']
SHOT_CLOCK (numeric, 5808 distinct): ['24.0', '11.0', '12.0', '14.0', '13.0', '10.0', '9.0', '15.0', '16.0', '8.0']
DRIBBLES (numeric, 33 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
TOUCH_TIME (numeric, 313 distinct): ['0.8', '0.9', '1.0', '0.7', '1.1', '0.6', '1.2', '0.0', '1.4', '1.3']
SHOT_DIST (numeric, 448 distinct): ['24.4', '24.5', '24.6', '24.7', '24.3', '24.2', '24.8', '24.9', '25.0', '24.1']
PTS_TYPE (numeric, 2 distinct): ['2', '3']
SHOT_RESULT (nominal, 2 distinct): ['missed', 'made']
CLOSEST_DEFENDER (nominal, 473 distinct): ['Ibaka, Serge', 'Jordan, DeAndre', 'Gasol, Pau', 'Green, Draymond', 'Millsap, Paul', 'Chandler, Tyson', 'Vucevic, Nikola', 'Frye, Channing', 'Love, Kevin', 'Gortat, Marcin']
CLOSEST_DEFENDER_PLAYER_ID (numeric, 474 distinct): ['201586.0', '201599.0', '2200.0', '203110.0', '200794.0', '2199.0', '202696.0', '101112.0', '201567.0', '101162.0']
CLOSE_DEF_DIST (numeric, 299 distinct): ['2.2', '1.9', '2.1', '2.3', '2.0', '2.4', '2.5', '2.6', '2.7', '2.8']
FGM (numeric, 2 distinct): ['0', '1']
PTS (numeric, 3 distinct): ['0', '2', '3']
player_name (nominal, 281 distinct): ['james harden', 'mnta ellis', 'lamarcus aldridge', 'damian lillard', 'lebron james', 'klay thompson', 'russell westbrook', 'stephen curry', 'kyrie irving', 'tyreke evans']
player_id (numeric, 281 distinct): ['201935.0', '101145.0', '200746.0', '203081.0', '2544.0', '202691.0', '201566.0', '201939.0', '202681.0', '201936.0']
'''

CONTEXT = "NBA 2014-2015 Season Shots Statistics"
TARGET = CuratedTarget(raw_name="SHOT_RESULT", task_type=SupervisedTask.BINARY)
# FGM and PTS are leakages. The others are meaningless IDs
COLS_TO_DROP = ["FGM", "PTS", "GAME_ID", "CLOSEST_DEFENDER_PLAYER_ID", "player_id"]
FEATURES = [CuratedFeature(raw_name="MATCHUP", new_name="Match Details and Teams"),
            CuratedFeature(raw_name="LOCATION", value_mapping={'A': 'Away', 'H': 'Home'}),
            CuratedFeature(raw_name="W", new_name="Basketball Match Result", value_mapping={'W': 'Win', 'L': 'Loss'}),
            CuratedFeature(raw_name="FINAL_MARGIN", new_name="Result Final Margin"),
            CuratedFeature(raw_name="PERIOD", new_name="NBA Game Period"),
            CuratedFeature(raw_name="TOUCH_TIME", new_name="Touch Time (seconds)"),
            CuratedFeature(raw_name="SHOT_DIST", new_name="Shot Distance (feet)"),
            CuratedFeature(raw_name="PTS_TYPE", new_name="Points Type", value_mapping={'2': '2-point', '3': '3-point'}),
            CuratedFeature(raw_name="CLOSE_DEF_DIST", new_name="Closest Defender Distance (feet)"),
            ]
