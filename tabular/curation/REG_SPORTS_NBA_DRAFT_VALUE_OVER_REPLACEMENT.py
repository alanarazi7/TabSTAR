from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mattop/nba-draft-basketball-player-data-19892021/nbaplayersdraft.csv
====
Examples: 1922
====
URL: https://www.kaggle.com/mattop/nba-draft-basketball-player-data-19892021/nbaplayersdraft.csv
====
Description: 
NBA Draft Basketball Player Data 1989-2021
NBA Draft Data with Player Performance 1989-2021

About Dataset
The dataset contains all NBA Draft picks from 1989-2021. Dataset consists of year, overall pick and player data.

Notable players: LeBron James, Kobe Bryant, Derrick Rose, Dirk Nowitzki, Carmelo Anthony, Stephen Curry, Paul Pierce, Kevin Durant, Shaq, Vince Carter, Allen Iverson

Data from: https://www.basketball-reference.com/draft/

====
Features:

id (int64, 1922 distinct): ['1', '1442', '1290', '1289', '1288', '1287', '1286', '1285', '1284', '1283']
year (int64, 33 distinct): ['2005', '2006', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013']
rank (int64, 60 distinct): ['1', '42', '2', '32', '33', '34', '35', '36', '37', '38']
overall_pick (int64, 60 distinct): ['1', '42', '2', '32', '33', '34', '35', '36', '37', '38']
team (object, 38 distinct): ['PHI', 'ATL', 'MIN', 'PHO', 'BOS', 'CHI', 'SAC', 'UTA', 'MIL', 'ORL']
player (object, 1916 distinct): ['Justin Jackson', 'Corey Brewer', 'Marcus Thornton', 'Marcus Williams', 'Michael Smith', 'Dee Brown', 'Nikola Mirotić', 'Donatas Motiejūnas', 'Nolan Smith', 'Kenneth Faried']
college (object, 237 distinct): ['Kentucky', 'Duke', 'Arizona', 'UNC', 'UCLA', 'Kansas', 'UConn', 'Syracuse', 'Michigan', 'Texas']
years_active (float64, 22 distinct): ['2.0', '1.0', '3.0', '4.0', '5.0', '6.0', '10.0', '8.0', '11.0', '13.0']
games (float64, 756 distinct): ['13.0', '2.0', '22.0', '36.0', '6.0', '51.0', '16.0', '64.0', '103.0', '23.0']
minutes_played (float64, 1522 distinct): ['28.0', '58.0', '32.0', '120.0', '39.0', '303.0', '92.0', '74.0', '38.0', '6.0']
points (float64, 1338 distinct): ['0.0', '18.0', '4.0', '2.0', '204.0', '14.0', '37.0', '22.0', '55.0', '3.0']
total_rebounds (float64, 1149 distinct): ['3.0', '0.0', '4.0', '2.0', '7.0', '6.0', '14.0', '31.0', '15.0', '19.0']
assists (float64, 920 distinct): ['0.0', '3.0', '2.0', '1.0', '5.0', '4.0', '26.0', '19.0', '17.0', '7.0']
field_goal_percentage (float64, 318 distinct): ['0.417', '0.425', '0.435', '0.434', '0.46', '0.437', '0.411', '0.333', '0.404', '0.412']
3_point_percentage (float64, 287 distinct): ['0.0', '0.25', '0.333', '0.2', '0.36', '0.349', '0.357', '0.361', '0.34', '0.286']
free_throw_percentage (float64, 394 distinct): ['0.5', '0.667', '0.75', '1.0', '0.714', '0.743', '0.769', '0.78', '0.799', '0.81']
average_minutes_played (float64, 350 distinct): ['9.9', '20.9', '22.2', '16.0', '13.9', '20.4', '21.9', '3.0', '15.2', '18.2']
points_per_game (float64, 220 distinct): ['3.3', '3.2', '3.0', '2.7', '2.8', '2.3', '2.2', '2.0', '5.9', '3.6']
average_total_rebounds (float64, 111 distinct): ['1.0', '2.2', '1.8', '1.9', '3.1', '2.8', '1.4', '1.7', '2.0', '2.6']
average_assists (float64, 85 distinct): ['0.7', '0.5', '0.8', '0.3', '0.6', '0.4', '0.2', '0.9', '1.0', '1.1']
win_shares (float64, 544 distinct): ['0.0', '-0.1', '0.1', '0.2', '-0.2', '0.5', '0.3', '0.4', '0.8', '1.0']
win_shares_per_48_minutes (float64, 317 distinct): ['0.078', '0.058', '0.068', '0.086', '0.079', '0.093', '0.069', '0.075', '0.104', '0.051']
box_plus_minus (float64, 192 distinct): ['-2.0', '-0.4', '-2.1', '-2.7', '-1.8', '-2.8', '-0.9', '-1.6', '-1.9', '-2.6']
value_over_replacement (float64, 320 distinct): ['0.0', '-0.1', '-0.2', '-0.3', '-0.4', '-0.6', '-0.5', '0.1', '-0.8', '-0.7']
'''

CONTEXT = "NBA Draft Basketball Player Data 1989-2021"
TARGET = CuratedTarget(raw_name="value_over_replacement", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id"]
FEATURES = []

DESCRIPTION = '''
NBA Draft Basketball Player Data 1989-2021
NBA Draft Data with Player Performance 1989-2021

About Dataset
The dataset contains all NBA Draft picks from 1989-2021. Dataset consists of year, overall pick and player data.

Notable players: LeBron James, Kobe Bryant, Derrick Rose, Dirk Nowitzki, Carmelo Anthony, Stephen Curry, Paul Pierce, Kevin Durant, Shaq, Vince Carter, Allen Iverson

Data from: https://www.basketball-reference.com/draft/
'''