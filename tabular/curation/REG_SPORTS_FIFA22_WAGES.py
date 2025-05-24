from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: fifa
====
Examples: 19178
====
URL: https://www.openml.org/search?type=data&id=45012
====
Description: **Data Description**

The datasets provided include the players data for the Career Mode from FIFA 22.
It only includes male football players with known wage.

The goal is to predict the wage of the player based on his attributes.

**Attribute Description**

The features describe self-explanatory properties and skills of the player.
====
Target Variable: wage_eur (numeric, 133 distinct): ['2000.0', '500.0', '1000.0', '3000.0', '4000.0', '5000.0', '6000.0', '7000.0', '8000.0', '9000.0']
====
Features:

age (numeric, 29 distinct): ['21', '22', '24', '25', '23', '20', '27', '26', '29', '28']
height_cm (numeric, 49 distinct): ['180', '185', '178', '183', '175', '188', '182', '184', '186', '181']
weight_kg (numeric, 58 distinct): ['70', '75', '80', '72', '73', '74', '78', '76', '77', '68']
nationality_name (nominal, 163 distinct): ['England', 'Germany', 'Spain', 'France', 'Argentina', 'Brazil', 'Japan', 'Netherlands', 'United States', 'Poland']
overall (numeric, 47 distinct): ['65', '67', '64', '66', '63', '68', '62', '69', '70', '60']
potential (numeric, 46 distinct): ['72', '70', '68', '69', '71', '73', '67', '74', '66', '75']
attacking_crossing (numeric, 88 distinct): ['58', '60', '65', '64', '62', '59', '55', '63', '56', '57']
attacking_finishing (numeric, 94 distinct): ['58', '60', '59', '55', '65', '64', '62', '63', '61', '52']
attacking_heading_accuracy (numeric, 89 distinct): ['58', '55', '62', '59', '60', '64', '65', '56', '54', '57']
attacking_short_passing (numeric, 86 distinct): ['64', '65', '66', '62', '63', '67', '60', '68', '61', '58']
attacking_volleys (numeric, 88 distinct): ['55', '59', '48', '49', '45', '42', '52', '53', '41', '54']
skill_dribbling (numeric, 92 distinct): ['65', '64', '63', '62', '66', '68', '60', '67', '61', '70']
skill_curve (numeric, 89 distinct): ['48', '45', '55', '58', '60', '59', '49', '52', '42', '63']
skill_fk_accuracy (numeric, 90 distinct): ['35', '42', '32', '38', '40', '39', '31', '45', '30', '41']
skill_long_passing (numeric, 85 distinct): ['62', '60', '58', '65', '55', '64', '59', '63', '61', '57']
skill_ball_control (numeric, 88 distinct): ['65', '64', '63', '62', '66', '68', '60', '70', '67', '61']
movement_acceleration (numeric, 84 distinct): ['68', '69', '67', '70', '73', '72', '66', '74', '65', '71']
movement_sprint_speed (numeric, 83 distinct): ['68', '69', '67', '65', '70', '66', '73', '72', '71', '75']
movement_agility (numeric, 79 distinct): ['70', '72', '65', '71', '68', '66', '73', '67', '75', '69']
movement_reactions (numeric, 67 distinct): ['60', '62', '58', '65', '64', '63', '67', '59', '66', '55']
movement_balance (numeric, 79 distinct): ['70', '68', '65', '71', '72', '67', '66', '69', '73', '64']
defending_standing_tackle (numeric, 88 distinct): ['65', '64', '63', '62', '66', '67', '68', '70', '60', '61']
defending_sliding_tackle (numeric, 88 distinct): ['62', '64', '65', '60', '63', '61', '58', '13', '59', '14']
goalkeeping_diving (numeric, 71 distinct): ['8', '14', '7', '9', '11', '13', '10', '12', '6', '15']
goalkeeping_handling (numeric, 69 distinct): ['10', '12', '9', '8', '14', '7', '13', '11', '6', '15']
goalkeeping_kicking (numeric, 79 distinct): ['9', '12', '13', '7', '14', '10', '11', '8', '6', '15']
goalkeeping_positioning (numeric, 77 distinct): ['8', '10', '7', '11', '12', '9', '14', '13', '6', '15']
goalkeeping_reflexes (numeric, 70 distinct): ['9', '11', '7', '8', '13', '10', '14', '12', '6', '15']
'''

CONTEXT = "FIFA 2022 Players Wages"
TARGET = CuratedTarget(raw_name="wage_eur", new_name="Wage in Euros", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []