from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: baseball-hitter
====
Examples: 263
====
URL: https://www.openml.org/search?type=data&id=525
====
Description: **Author**: Lorraine Denby    
**Source**: [StatLib](http://lib.stat.cmu.edu/datasets/)  
**Please cite**:   

**Analysis of Baseball Salary Data: Hitters**
This analysis describes and summarizes the relationships between 1987 salaries of major league baseball players and the player's performance. The salary data were taken from Sports Illustrated, April 20, 1987. The salary of any player not included in that article is listed as an NA. The 1986 and career statistics were taken from The 1987 Baseball Encyclopedia Update published by Collier  Books,  Macmillan  Publishing  Company, New York. The team attendance figures were obtained from the Elias Sports Bureau, personal conversation.  

The data consist of data on the regular and leading substitute hitters in 1986 of North American Major League Baseball players. 

### Attribute information  
There is one line per hitter. The variables are:
hitter's name,  
number of times at bat in 1986,  
number of hits in 1986,  
number of home runs in 1986,  
number of runs in 1986,  
number of runs batted in in 1986,  
number of walks in 1986,  
number of years in the major leagues,  
number of times at bat during his career,  
number of hits during his career,  
number of home runs during his career,  
number of runs during his career,  
number of runs batted in during his career,  
number of walks during his career,  
player's league at the end of 1986,  
player's division at the end of 1986,  
player's team at the end of 1986,  
player's position(s) in 1986,  
number of put outs in 1986,  
number of assists in 1986,  
number of errors in 1986,  
1987 annual salary on opening day in thousands of dollars,  
player's league at the beginning of 1987,  
player's team at the beginning of 1987.
====
Target Variable: 1987_annual_salary_on_opening_day_in_thousands_of_dollars (numeric, 150 distinct): ['750.0', '100.0', '250.0', '90.0', '700.0', '450.0', '70.0', '75.0', '300.0', '850.0']
====
Features:

number_of_times_at_bat_in_1986 (numeric, 209 distinct): ['315', '283', '216', '591', '288', '205', '327', '490', '441', '155']
number_of_hits_in_1986 (numeric, 130 distinct): ['70', '163', '76', '101', '68', '103', '53', '152', '113', '81']
number_of_home_runs_in_1986 (numeric, 35 distinct): ['5', '4', '8', '6', '3', '7', '9', '2', '13', '0']
number_of_runs_in_1986 (numeric, 92 distinct): ['42', '67', '50', '89', '26', '48', '32', '24', '61', '54']
number_of_runs_batted_in_in_1986 (numeric, 94 distinct): ['29', '25', '36', '23', '60', '47', '44', '45', '33', '58']
number_of_walks_in_1986 (numeric, 87 distinct): ['22', '30', '21', '39', '32', '18', '52', '15', '26', '62']
number_of_years_in_the_major_leagues (numeric, 21 distinct): ['4', '6', '5', '3', '2', '7', '1', '14', '9', '10']
number_of_times_at_bat_during_his_career (numeric, 257 distinct): ['1350', '216', '1928', '2331', '711', '1258', '1968', '498', '41', '4739']
number_of_hits_during_his_career (numeric, 241 distinct): ['160', '880', '68', '715', '113', '151', '78', '210', '54', '149']
number_of_home_runs_during_his_career (numeric, 129 distinct): ['16', '12', '32', '24', '2', '1', '7', '3', '6', '9']
number_of_runs_during_his_career (numeric, 226 distinct): ['102', '32', '181', '80', '352', '99', '450', '34', '20', '27']
number_of_runs_batted_in_during_his_career (numeric, 226 distinct): ['69', '46', '80', '37', '32', '491', '167', '163', '230', '342']
number_of_walks_during_his_career (numeric, 207 distinct): ['55', '91', '45', '174', '71', '198', '155', '136', '24', '33']
players_league_at_the_end_of_1986 (nominal, 2 distinct): ['A', 'N']
players_division_at_the_end_of_1986 (nominal, 2 distinct): ['W', 'E']
players_team_at_the_end_of_1986 (nominal, 24 distinct): ['Chi.', 'N.Y.', 'S.F.', 'S.D.', 'Min.', 'Tex.', 'Cle.', 'Det.', 'St.L.', 'Sea.']
players_position(s)_in_1986 (nominal, 23 distinct): ['C', '3B', 'SS', '2B', '1B', 'CF', 'RF', 'OF', 'LF', 'DH']
number_of_put_outs_in_1986 (numeric, 199 distinct): ['0', '303', '102', '276', '211', '280', '325', '121', '226', '203']
number_of_assists_in_1986 (numeric, 145 distinct): ['0', '9', '6', '7', '5', '8', '4', '2', '3', '10']
number_of_errors_in_1986 (numeric, 29 distinct): ['3', '6', '4', '5', '2', '8', '0', '7', '9', '16']
players_league_at_the_beginning_of_1987 (nominal, 2 distinct): ['A', 'N']
players_team_at_the_beginning_of_1987 (nominal, 24 distinct): ['N.Y.', 'Chi.', 'S.F.', 'Pit.', 'Min.', 'Cle.', 'Det.', 'Phi.', 'Hou.', 'Bal.']
'''

TEAM_MAP = {"Atl.": "Atlanta", "Bal.": "Baltimore", "Bos.": "Boston", "Cal.": "California", "Chi.": "Chicago",
            "Cin.": "Cincinnati", "Cle.": "Cleveland", "Det.": "Detroit", "Hou.": "Houston",
            "K.C.": "Kansas City", "L.A.": "Los Angeles", "Mil.": "Milwaukee", "Min.": "Minnesota",
            "Mon.": "Montreal", "N.Y.": "New York", "Oak.": "Oakland", "Phi.": "Philadelphia",
            "Pit.": "Pittsburgh", "S.D.": "San Diego", "S.F.": "San Francisco", "Sea.": "Seattle",
            "St.L.": "St. Louis", "Tex.": "Texas", "Tor.": "Toronto"}
LEAGUE_MAP = {"A": "American League", "N": "National League"}
DIVISION_MAP = {"E": "East", "W": "West"}


CONTEXT = "Major League Baseball Hitters 1987"
TARGET = CuratedTarget(raw_name="1987_annual_salary_on_opening_day_in_thousands_of_dollars",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name="players_league_at_the_end_of_1986", value_mapping=LEAGUE_MAP),
    CuratedFeature(raw_name="players_division_at_the_end_of_1986", value_mapping=DIVISION_MAP),
    CuratedFeature(raw_name="players_team_at_the_end_of_1986", value_mapping=TEAM_MAP),
    CuratedFeature(raw_name="players_league_at_the_beginning_of_1987", value_mapping=LEAGUE_MAP),
    CuratedFeature(raw_name="players_team_at_the_beginning_of_1987", value_mapping=TEAM_MAP),
    ]