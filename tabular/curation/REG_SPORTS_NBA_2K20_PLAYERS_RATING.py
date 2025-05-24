from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: NBA-2k20-player-dataset
====
Examples: 439
====
URL: https://www.openml.org/search?type=data&id=43420
====
Description: Context
NBA 2k20 analysis.
Content
Detailed attributes for players registered in the NBA2k20.
Acknowledgements
Data scraped from https://hoopshype.com/nba2k/. Additional data about countries and drafts scraped from Wikipedia.
Inspiration
Inspired from this dataset: https://www.kaggle.com/karangadiya/fifa19
====
Features:

full_name (string, 429 distinct): ['LeBron James', 'Giannis Antetokounmpo', 'Kevin Durant', 'James Harden', 'Stephen Curry', 'Anthony Davis', 'Luka Doncic', 'Damian Lillard', 'Joel Embiid', 'Kawhi Leonard']
rating (numeric, 31 distinct): ['72', '73', '76', '74', '75', '77', '71', '78', '70', '79']
jersey (string, 52 distinct): ['0', '3', '5', '11', '9', '1', '8', '7', '2', '22']
team (string, 31 distinct): ['Milwaukee Bucks', 'Los Angeles Lakers', 'Dallas Mavericks', 'Brooklyn Nets', 'Phoenix Suns', 'New York Knicks', 'Chicago Bulls', 'Orlando Magic', 'Philadelphia 76ers', 'New Orleans Pelicans']
position (string, 7 distinct): ['G', 'F', 'C', 'F-C', 'G-F', 'F-G', 'C-F']
b_day (string, 415 distinct): ['12/30/84', '09/14/89', '03/25/86', '08/23/90', '07/15/91', '05/30/92', '02/10/95', '04/01/88', '07/20/91', '09/19/96']
height (string, 20 distinct): ['6-6 / 1.98', '6-8 / 2.03', '6-5 / 1.96', '6-7 / 2.01', '6-10 / 2.08', '6-3 / 1.91', '6-4 / 1.93', '6-9 / 2.06', '6-11 / 2.11', '7-0 / 2.13']
weight (string, 84 distinct): ['215 lbs. / 97.5 kg.', '190 lbs. / 86.2 kg.', '225 lbs. / 102.1 kg.', '220 lbs. / 99.8 kg.', '200 lbs. / 90.7 kg.', '205 lbs. / 93 kg.', '195 lbs. / 88.5 kg.', '180 lbs. / 81.6 kg.', '210 lbs. / 95.3 kg.', '230 lbs. / 104.3 kg.']
salary (string, 316 distinct): ['1416852', '898310', '1618520', '4767000', '2564753', '79568', '8000000', '32742000', '1737145', '27285000']
country (string, 39 distinct): ['USA', 'Canada', 'Australia', 'France', 'Germany', 'Croatia', 'Spain', 'Serbia', 'Greece', 'Italy']
draft_year (numeric, 18 distinct): ['2018', '2017', '2019', '2016', '2015', '2014', '2013', '2012', '2011', '2009']
draft_round (string, 3 distinct): ['1', '2', 'Undrafted']
draft_peak (string, 57 distinct): ['Undrafted', '2', '3', '1', '9', '7', '15', '5', '4', '12']
college (string, 110 distinct): ['Kentucky', 'Duke', 'North Carolina', 'Texas', 'UCLA', 'Kansas', 'Arizona', 'Villanova', 'Indiana', 'Michigan']
version (string, 2 distinct): ['NBA2k20', 'NBA2k21']
'''

CONTEXT = "NBA Players in NBA 2k20 Game"
# I randomly decided the target variable should be the player rating, this was not the original dataset.
TARGET = CuratedTarget(raw_name="rating", task_type=SupervisedTask.REGRESSION)
FEATURES = [
            CuratedFeature(raw_name="jersey", new_name="Player Jersey Number", feat_type=FeatureType.CATEGORICAL),
            CuratedFeature(raw_name="position", new_name="Player Position",
                            value_mapping={'G': 'Guard', 'F': 'Forward', 'C': 'Center', 'F-C': 'Forward-Center',
                                           'G-F': 'Guard-Forward', 'F-G': 'Forward-Guard', 'C-F': 'Center-Forward'}),
            CuratedFeature(raw_name="b_day", new_name="Player Birthday", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="draft_peak", new_name="Player Draft Peak"),
            CuratedFeature(raw_name="version", new_name="NBA 2K Version"),
            ]
COLS_TO_DROP = []
