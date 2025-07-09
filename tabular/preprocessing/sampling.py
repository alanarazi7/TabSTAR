from random import sample
from typing import Tuple

from pandas import DataFrame, Series

from tabstar_paper.pretraining.datasets import MAX_PRETRAIN_FEATURES, MAX_PRETRAIN_EXAMPLES
from tabular.datasets.manual_curation_obj import CuratedDataset


def deprecated__subsample_big_datasets(x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    if len(y) < MAX_PRETRAIN_EXAMPLES:
        return x, y
    indices = y.sample(n=MAX_PRETRAIN_EXAMPLES).index
    return x.loc[indices], y.loc[indices]

def deprecated__downsample_multiple_features(x: DataFrame, curation: CuratedDataset) -> Tuple[DataFrame, CuratedDataset]:
    # PRETRAIN REFACTOR: remove this function after new flow works, only relevant for pretraining.
    # TODO: This is EXTREMELY naive, we could use a more sophisticated way to avoid losing important features
    if len(x.columns) <= MAX_PRETRAIN_FEATURES:
        return x, curation
    print(f"🎲 Downsampling features for {curation.name} from {len(x.columns)} to {MAX_PRETRAIN_FEATURES}")
    columns = list(x.columns)
    chosen_columns = sample(columns, k=MAX_PRETRAIN_FEATURES)
    x = x[chosen_columns]
    curation.features = [f for f in curation.features if f.raw_name in chosen_columns]
    return x, curation