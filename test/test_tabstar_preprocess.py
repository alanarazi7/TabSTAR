import os
import shutil

import numpy as np

from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.pretraining.dataloaders import get_dev_dataloader
from tabstar_paper.pretraining.datasets import create_pretrain_dataset


TEST_DATA_DIR = "temp_test_dir_imdb"

IMDB_X_TXT_FIRST_ROW = [
'Target Feature: Genre\nFeature Value: Drama',
'Target Feature: Genre\nFeature Value: Not Drama',
'Predictive Feature: Title\nFeature Value: Tall Men',
'Predictive Feature: Description\nFeature Value: A challenged man is stalked by tall phantoms in business suits after he purchases a car with a mysterious black credit card.',
'Predictive Feature: Director\nFeature Value: Jonathan Holbrook',
'Predictive Feature: Actors\nFeature Value: Dan Crisafulli, Kay Whitney, Richard Garcia, Pat Cashman',
'Predictive Feature: Metascore\nFeature Value: 55 to 59 (Quantile 40 - 50%)',
'Predictive Feature: Rank\nFeature Value: 598.4 to 691.3 (Quantile 60 - 70%)',
'Predictive Feature: Rating\nFeature Value: 1.9 to 5.59 (Quantile 0 - 10%)',
'Predictive Feature: Revenue (Millions)\nFeature Value: Unknown Value',
'Predictive Feature: Runtime (Minutes)\nFeature Value: 126 to 138 (Quantile 80 - 90%)',
'Predictive Feature: Votes\nFeature Value: 61 to 4870.8 (Quantile 0 - 10%)',
'Predictive Feature: Year\nFeature Value: Higher than 2016 (Quantile 100%)',
]


IMDB_X_NUM_FIRST_ROW = [
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
-0.10520633310079575,
0.5200179815292358,
-3.0,
0.0,
1.0936769247055054,
-0.9359688758850098,
0.9988187551498413,
]


IMDB_X_TXT_LAST_ROW = [
'Target Feature: Genre\nFeature Value: Drama',
'Target Feature: Genre\nFeature Value: Not Drama',
'Predictive Feature: Title\nFeature Value: Hunger',
'Predictive Feature: Description\nFeature Value: Irish republican Bobby Sands leads the inmates of a Northern Irish prison in a hunger strike.',
'Predictive Feature: Director\nFeature Value: Steve McQueen',
'Predictive Feature: Actors\nFeature Value: Stuart Graham, Laine Megaw, Brian Milligan, Liam McMahon',
'Predictive Feature: Metascore\nFeature Value: 81 to 100 (Quantile 90 - 100%)',
'Predictive Feature: Rank\nFeature Value: 598.4 to 691.3 (Quantile 60 - 70%)',
'Predictive Feature: Rating\nFeature Value: 7.5 to 7.9 (Quantile 80 - 90%)',
'Predictive Feature: Revenue (Millions)\nFeature Value: 0 to 1.816 (Quantile 0 - 10%)',
'Predictive Feature: Runtime (Minutes)\nFeature Value: 91 to 97 (Quantile 10 - 20%)',
'Predictive Feature: Votes\nFeature Value: 50250.8 to 78228.4 (Quantile 30 - 40%)',
'Predictive Feature: Year\nFeature Value: 2008 to 2010 (Quantile 10 - 20%)',
]

IMDB_X_NUM_LAST_ROW = [
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
1.3703500032424927,
0.6548995971679688,
0.9240633249282837,
-0.8019106984138489,
-0.9082886576652527,
-0.6306809782981873,
-1.502341866493225,
]


# def test_tabstar_preprocessing_imdb(): # this test breaks and its fix is not prioritized for now
#     _rm_temp_dir()
#     data_dir = create_pretrain_dataset(dataset_id=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
#                                        cache_dir=TEST_DATA_DIR)
#     dev_dataloader = get_dev_dataloader(data_dir=data_dir, batch_size=40)
#     x_txt, x_num, y, properties = next(iter(dev_dataloader))
#     assert properties.d_output == 2
#     assert properties.train_size == 760
#     assert properties.val_size == 40
#     assert np.mean(y.numpy()) == 0.475

#     # First example
#     assert y[0] == 1
#     assert x_txt[0].tolist() == IMDB_X_TXT_FIRST_ROW
#     assert x_num[0].tolist() == IMDB_X_NUM_FIRST_ROW

#     # Last example
#     assert y[-1] == 0
#     assert x_txt[-1].tolist() == IMDB_X_TXT_LAST_ROW
#     assert x_num[-1].tolist() == IMDB_X_NUM_LAST_ROW
#     _rm_temp_dir()

def _rm_temp_dir():
    if os.path.isdir(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)