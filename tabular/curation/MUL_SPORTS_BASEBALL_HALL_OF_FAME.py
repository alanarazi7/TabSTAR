from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: baseball
====
Examples: 1340
====
URL: https://www.openml.org/search?type=data&id=185
====
Description: Database of baseball players and play statistics, including 'Games_played', 'At_bats', 'Runs', 'Hits', 'Doubles', 'Triples', 'Home_runs', 'RBIs', 'Walks', 'Strikeouts', 'Batting_average', 'On_base_pct', 'Slugging_pct' and 'Fielding_ave' 

Notes:  
* Quotes, Single-Quotes and Backslashes were removed, Blanks replaced with Underscores
* Player is an identifier that should be ignored when modelling the data
====
Target Variable: Hall_of_Fame (nominal, 3 distinct): ['0', '2', '1']
====
Features:

Number_seasons (numeric, 17 distinct): ['10', '11', '12', '13', '15', '14', '17', '16', '18', '19']
Games_played (numeric, 981 distinct): ['835', '1435', '989', '918', '1853', '1421', '1806', '730', '813', '1035']
At_bats (numeric, 1239 distinct): ['4101', '4019', '4553', '4255', '4992', '2877', '1963', '3172', '4603', '3421']
Runs (numeric, 812 distinct): ['331', '357', '562', '676', '163', '237', '245', '598', '552', '146']
Hits (numeric, 999 distinct): ['972', '1103', '786', '833', '1415', '1252', '813', '1122', '877', '881']
Doubles (numeric, 418 distinct): ['162', '128', '91', '216', '71', '172', '196', '190', '147', '244']
Triples (numeric, 180 distinct): ['12', '28', '33', '26', '11', '10', '35', '17', '31', '23']
Home_runs (numeric, 291 distinct): ['9', '7', '21', '18', '22', '6', '27', '13', '20', '14']
RBIs (numeric, 795 distinct): ['591', '391', '501', '525', '272', '417', '438', '513', '329', '229']
Walks (numeric, 712 distinct): ['299', '383', '190', '369', '351', '309', '143', '333', '331', '188']
Strikeouts (numeric, 723 distinct): ['218.0', '453.0', '243.0', '312.0', '382.0', '345.0', '361.0', '357.0', '251.0', '329.0']
Batting_average (numeric, 143 distinct): ['0.271', '0.267', '0.261', '0.262', '0.254', '0.265', '0.273', '0.268', '0.264', '0.257']
On_base_pct (numeric, 176 distinct): ['0.34', '0.323', '0.333', '0.329', '0.318', '0.346', '0.351', '0.316', '0.319', '0.331']
Slugging_pct (numeric, 274 distinct): ['0.351', '0.355', '0.367', '0.379', '0.327', '0.343', '0.377', '0.364', '0.358', '0.333']
Fielding_ave (numeric, 125 distinct): ['0.984', '0.982', '0.983', '0.98', '0.981', '0.99', '0.973', '0.974', '0.985', '0.978']
Position (nominal, 7 distinct): ['Outfield', 'Catcher', 'Shortstop', 'Second_base', 'Third_base', 'First_base', 'Designated_hitter']
'''

CONTEXT = "Baseball Players and Play Statistics"
TARGET = CuratedTarget(raw_name="Hall_of_Fame", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'0': 'Not inducted', '1': 'Inducted', '2': 'Nominated'})
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name="RBIs", new_name="Runs Batted In"),
    CuratedFeature(raw_name="On_base_pct", new_name="On-Base Percentage"),
    CuratedFeature(raw_name="Slugging_pct", new_name="Slugging Percentage"),
    CuratedFeature(raw_name="Fielding_ave", new_name="Fielding Average"),
    ]
