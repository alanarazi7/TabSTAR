import os
import shutil

from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.pretraining.dataloaders import get_dev_dataloader
from tabstar_paper.pretraining.datasets import create_pretrain_dataset


TEST_DATA_DIR = "temp_test_dir_imdb"

IMDB_Y_DEV_BATCH = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
IMDB_X_NUM_FIRST_ROW = [0.0, 0.0, 0.0116, -0.0259, -0.0412, -0.9315, 0.9855, 0.6862, -0.4594, 0.0, 0.0, 0.0, 0.0]
IMDB_X_TXT_FIRST_ROW = ['Target Feature: Genre_is_Drama\nFeature Value: Drama',
                        'Target Feature: Genre_is_Drama\nFeature Value: Not Drama',
                        'Predictive Feature: Rank\nFeature Value: 494.5 to 598.4 (Quantile 50 - 60%)',
                        'Predictive Feature: Revenue (Millions)\nFeature Value: 65.008 to 94.472 (Quantile 60 - 70%)',
                        'Predictive Feature: Votes\nFeature Value: 153848.6 to 205423.9 (Quantile 60 - 70%)',
                        'Predictive Feature: Metascore\nFeature Value: 43 to 49 (Quantile 20 - 30%)',
                        'Predictive Feature: Runtime (Minutes)\nFeature Value: 126 to 138 (Quantile 80 - 90%)',
                        'Predictive Feature: Year\nFeature Value: 2015 to 2016 (Quantile 60 - 70%)',
                        'Predictive Feature: Rating\nFeature Value: 6.3 to 6.6 (Quantile 30 - 40%)',
                        'Predictive Feature: Director\nFeature Value: Wes Ball',
                        'Predictive Feature: Title\nFeature Value: Maze Runner: The Scorch Trials',
                        "Predictive Feature: Actors\nFeature Value: Dylan O'Brien, Kaya Scodelario, Thomas Brodie-Sangster,Giancarlo Esposito",
                        'Predictive Feature: Description\nFeature Value: After having escaped the Maze, the Gladers now face a new set of challenges on the open roads of a desolate landscape filled with unimaginable obstacles.'
                        ]

X_TXT_STAM = ['Target Feature: Genre\nFeature Value: Drama', 'Target Feature: Genre\nFeature Value: Not Drama', 'Predictive Feature: Title\nFeature Value: Tall Men', 'Predictive Feature: Description\nFeature Value: A challenged man is stalked by tall phantoms in business suits after he purchases a car with a mysterious black credit card.', 'Predictive Feature: Director\nFeature Value: Jonathan Holbrook', 'Predictive Feature: Actors\nFeature Value: Dan Crisafulli, Kay Whitney, Richard Garcia, Pat Cashman', 'Predictive Feature: Metascore\nFeature Value: 55 to 59 (Quantile 40 - 50%)', 'Predictive Feature: Rank\nFeature Value: 598.4 to 691.3 (Quantile 60 - 70%)', 'Predictive Feature: Rating\nFeature Value: 1.9 to 5.59 (Quantile 0 - 10%)', 'Predictive Feature: Revenue (Millions)\nFeature Value: Unknown Value', 'Predictive Feature: Runtime (Minutes)\nFeature Value: 126 to 138 (Quantile 80 - 90%)', 'Predictive Feature: Votes\nFeature Value: 61 to 4870.8 (Quantile 0 - 10%)', 'Predictive Feature: Year\nFeature Value: Higher than 2016 (Quantile 100%)']
X_NUM_STAM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1052, 0.52, -3.0, 0.0, 1.0937, -0.936, 0.9988]


# TODO: still working on replicating the 'tabular' repo
def test_tabstar_preprocessing_imdb():
    _rm_temp_dir()
    data_dir = create_pretrain_dataset(dataset_id=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
                                       cache_dir=TEST_DATA_DIR)
    dev_dataloader = get_dev_dataloader(data_dir=data_dir, batch_size=32)
    x_txt, x_num, y, properties = next(iter(dev_dataloader))
    assert properties.d_output == 2
    assert properties.train_size == 760
    assert properties.val_size == 40
    # assert y.tolist() == IMDB_Y_DEV_BATCH
    x_num_first_row_rounded = [round(x, 4) for x in x_num[0].tolist()]
    assert x_num_first_row_rounded == X_NUM_STAM
    x_txt_first_row = x_txt[0].tolist()
    assert x_txt_first_row == X_TXT_STAM
    _rm_temp_dir()

def _rm_temp_dir():
    if os.path.isdir(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)