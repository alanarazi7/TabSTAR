from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar_paper.datasets.downloading import download_dataset
from tabstar_paper.pretraining.hyperparameters import PRETRAIN_VAL_RATIO


@dataclass
class PretrainDatasetProperties:
    name: str
    d_output: int
    idx2text: Dict[int, str]


def create_pretrain_dataset(dataset_id: TabularDatasetID):
    train_data, val_data = prepare_pretrain_dataset(dataset_id=dataset_id)
    idx2text = {}
    fill_idx2text(train_data, idx2text=idx2text)
    fill_idx2text(val_data, idx2text=idx2text)
    properties = PretrainDatasetProperties(name=dataset_id.name, d_output=train_data.d_output, idx2text=idx2text)
    raise NotImplementedError("Store the prepared datasets in the HDF5 format.")


def prepare_pretrain_dataset(dataset_id: TabularDatasetID, verbose: bool = False) -> Tuple[TabSTARData, TabSTARData]:
    # TODO: extend this to be from a CSV
    dataset = download_dataset(dataset_id=dataset_id)
    raise_if_null_target(dataset.y)
    x_train, x_val, y_train, y_val = split_to_val(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls,
                                                  val_ratio=PRETRAIN_VAL_RATIO)
    preprocessor = TabSTARVerbalizer(is_cls=dataset.is_cls, verbose=verbose)
    preprocessor.fit(x_train, y_train)
    train_data = preprocessor.transform(x_train, y_train)
    val_data = preprocessor.transform(x_val, y_val)
    return train_data, val_data

def fill_idx2text(data: TabSTARData, idx2text: Dict[int, str]):
    assert isinstance(data.x_txt, np.ndarray), "data.x_txt must be a NumPy array"
    assert data.x_txt.ndim == 2, "data.x_txt must be 2D"
    all_texts = set(data.x_txt.ravel())
    for t in all_texts:
        if t not in idx2text:
            idx2text[len(idx2text)] = t
    text2idx = {t: i for i, t in idx2text.items()}
    data.x_txt = np.vectorize(lambda x: text2idx[x])(data.x_txt)
