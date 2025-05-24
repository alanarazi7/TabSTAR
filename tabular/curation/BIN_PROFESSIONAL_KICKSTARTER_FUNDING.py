from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: kick_starter_funding
====
Examples: 86502
====
URL: https://www.openml.org/search?type=data&id=46668
====
Description: Predict whether a proposed Kickstarter project will achieve funding goal based on text features
    like its title, description, numeric features like the amount of money requested, date posted, and
    categorical features like the country, currency, etc. This dataset represents a complex task where
    models must consider interactions between modalities to address a core question of Kickstarter's
    business: https://www.kaggle.com/codename007/funding-successful-projects

    We applied the next preprocessing for uploading to OpenML to properly quoted ARFF format: 
    dataset = dataset.map(lambda x: f'"{x}"' if isinstance(x, str) else x)
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: final_status (numeric, 2 distinct): ['0', '1']
====
Features:

name (string, 86314 distinct): ['"New EP/Music Development"', '"Canceled (Canceled)"', '"Aftermath"', '"Alone"', '"Requiem"', '"Wonderland"', '"The Journey"', '"New Album"', '"Bunny"', '"Debut Album"']
desc (string, 86006 distinct): ['"The Decentralized Dance Party was founded on the belief that Partying is an art form that has the power to change the world."', '"."', '"Breakout Artist Management has offered to work with and develop this project in the studio and we need your help!"', '"Rock Steady is the first manga about Rock. Rock Steady is an action/adventure manga, join us on this extraordinary journey"', '"The Impossible Girl redefines the rock tour. One show at a time."', '"After being blackmailed by her evil stepfather, a young woman must frame the man she has fallen for in order to avoid imprisonment."', '"Practically all of the original Dungeons & Dragons artwork that I created during my time at TSR was destroyed. Let\'s bring it back!"', '"Opera SmackDown turns the traditional vocal concert on its ear by combining elements of professional wrestling and Opera Competitions"', '"A group of American artists exchanging creative education with African artists!"', '"imagine roaming the world’s largest ocean year after year alone, calling out with the regularity of a metronome, & hearing no response."']
goal (numeric, 3060 distinct): ['5000.0', '10000.0', '1000.0', '3000.0', '2000.0', '2500.0', '15000.0', '500.0', '1500.0', '20000.0']
keywords (string, 86502 distinct): ['"cross-eyed-chicks"', '"flying-curves-pole-dance-studio-goddesses-empowere"', '"subterranea-a-3d-fantasy-role-playing-game"', '"the-whispers-of-water"', '"in-it-for-storms-new-album"', '"kinnis-watch-me-the-debut-single"', '"affected-the-cabin-an-oculus-rift-horror-experienc"', '"ali-chai-co"', '"the-tigers-cub"', '"the-legend-of-the-bat"']
disable_communication (nominal, 2 distinct): ['0', '1']
country (string, 10 distinct): ['"US"', '"GB"', '"CA"', '"AU"', '"NL"', '"NZ"', '"SE"', '"DK"', '"IE"', '"NO"']
currency (string, 9 distinct): ['"USD"', '"GBP"', '"CAD"', '"AUD"', '"EUR"', '"NZD"', '"SEK"', '"DKK"', '"NOK"']
deadline (numeric, 81433 distinct): ['1414814340', '1420088340', '1325393940', '1430452740', '1425185940', '1351742340', '1409543940', '1427860740', '1330577940', '1412135940']
created_at (numeric, 86460 distinct): ['1404826263', '1406053722', '1419183243', '1421797410', '1333308811', '1411406301', '1410551952', '1420417090', '1427162930', '1412122722']
'''

CONTEXT = "Kickstarter Funding Prediction"
TARGET = CuratedTarget(raw_name="final_status", new_name="Funding Status", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'Failed', '1': 'Successful'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="deadline", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="created_at", feat_type=FeatureType.DATE)]
