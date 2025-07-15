from os.path import join, exists
from random import sample
from typing import Tuple, Dict

import numpy as np

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar_paper.datasets.curation import TabularDataset
from tabstar_paper.datasets.downloading import download_dataset
from tabstar_paper.pretraining.hdf5 import save_pretrain_dataset, DatasetProperties, HDF5Dataset
from tabstar_paper.pretraining.hyperparameters import PRETRAIN_VAL_RATIO

MAX_PRETRAIN_EXAMPLES = 300_000
MAX_PRETRAIN_FEATURES = 200


def create_pretrain_dataset(dataset_id: TabularDatasetID, cache_dir: str = ".tabstar_datasets") -> str:
    data_dir = join(cache_dir, dataset_id.name.replace('/', '_'))
    if exists(join(data_dir, HDF5Dataset.PROPERTIES)):
        return data_dir
    train_data, val_data = prepare_pretrain_dataset(dataset_id=dataset_id)
    idx2text = fill_idx2text(train_data=train_data, val_data=val_data)
    properties = DatasetProperties(name=dataset_id.name, d_output=train_data.d_output, idx2text=idx2text,
                                   train_size=len(train_data), val_size=len(val_data))
    save_pretrain_dataset(data_dir=data_dir, train_data=train_data, val_data=val_data, properties=properties)
    return data_dir


def prepare_pretrain_dataset(dataset_id: TabularDatasetID, verbose: bool = False) -> Tuple[TabSTARData, TabSTARData]:
    # TODO: extend this to be from a CSV
    dataset = download_dataset(dataset_id=dataset_id)
    raise_if_null_target(dataset.y)
    _downsample_max_features(dataset)
    _downsample_max_examples(dataset)
    x_train, x_val, y_train, y_val = split_to_val(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls,
                                                  val_ratio=PRETRAIN_VAL_RATIO)
    preprocessor = TabSTARVerbalizer(is_cls=dataset.is_cls, verbose=verbose)
    preprocessor.fit(x_train, y_train)
    train_data = preprocessor.transform(x_train, y_train)
    val_data = preprocessor.transform(x_val, y_val)
    return train_data, val_data

def fill_idx2text(train_data: TabSTARData, val_data: TabSTARData) -> Dict[int, str]:
    # TODO: add test?
    assert isinstance(train_data.x_txt, np.ndarray) and isinstance(val_data.x_txt, np.ndarray)
    assert train_data.x_txt.ndim == 2 and val_data.x_txt.ndim == 2
    all_texts = set(train_data.x_txt.ravel()).union(set(val_data.x_txt.ravel()))
    idx2text = {i: t for i, t in enumerate(all_texts)}
    text2idx = {t: i for i, t in idx2text.items()}
    train_data.x_txt = np.vectorize(lambda x: text2idx[x])(train_data.x_txt)
    val_data.x_txt = np.vectorize(lambda x: text2idx[x])(val_data.x_txt)
    return idx2text


def _downsample_max_examples(dataset: TabularDataset):
    if len(dataset.y) < MAX_PRETRAIN_EXAMPLES:
        return
    print(f"🎲 Downsampling examples for {dataset.dataset_id} from {len(dataset.y)} to {MAX_PRETRAIN_EXAMPLES}")
    indices = dataset.y.sample(n=MAX_PRETRAIN_EXAMPLES).index
    dataset.x = dataset.x.loc[indices]
    dataset.y = dataset.y.loc[indices]


def _downsample_max_features(dataset: TabularDataset):
    # TODO: This is EXTREMELY naive, we could use a more sophisticated way to avoid losing important features
    if len(dataset.x.columns) <= MAX_PRETRAIN_FEATURES:
        return
    print(f"🎲 Downsampling features for {dataset.dataset_id} from {len(dataset.x.columns)} to {MAX_PRETRAIN_FEATURES}")
    columns = list(dataset.x.columns)
    chosen_columns = sample(columns, k=MAX_PRETRAIN_FEATURES)
    dataset.x = dataset.x[chosen_columns]
