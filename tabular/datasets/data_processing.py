from dataclasses import dataclass
from typing import Self, List, Dict, Optional

import numpy as np
import torch
from pandas import DataFrame, Series

from tabular.datasets.properties import DatasetProperties
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.splits import create_splits, DataSplit
from tabular.preprocessing.dates import process_dates
from tabular.preprocessing.trees.embeddings import transform_texts_to_embeddings
from tabular.preprocessing.objects import PreprocessingMethod, FeatureType
from tabular.preprocessing.target import process_y
from tabular.preprocessing.trees.categorical import process_x_cat_to_indices
from tabular.preprocessing.trees.numerical import fill_median_for_numerical_nulls
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.tabstar.preprocessing.textual import verbalize_x_txt
from tabular.tabstar.preprocessing.numerical import scale_x_num_and_add_categorical_bins
from tabular.tabstar.preprocessing.target import add_target_tokens


@dataclass
class TabularDataset:
    x: DataFrame
    y: Series
    splits: List[DataSplit]
    properties: DatasetProperties
    x_num: Optional[np.array] = None

    @classmethod
    def from_raw(cls, raw: RawDataset, processing: PreprocessingMethod, run_num: int, train_examples: int,
                 device: torch.device, number_verbalization: NumberVerbalization) -> Self:
        splits = create_splits(raw, run_num=run_num, train_examples=train_examples, processing=processing)
        targets = process_y(raw=raw, splits=splits, processing=processing)
        feat_cnt = {feat_type.value: len(names) for feat_type, names in raw.feature_types.items()}
        process_dates(raw=raw)
        assert not raw.feature_types[FeatureType.DATE]
        if processing == PreprocessingMethod.TABSTAR:
            return cls.for_tabstar(raw, splits=splits, targets=targets, feat_cnt=feat_cnt,
                                   number_verbalization=number_verbalization)
        elif processing in {PreprocessingMethod.TABPFNV2, PreprocessingMethod.CARTE}:
            return cls.from_processed(raw=raw, processing=processing, splits=splits, feat_cnt=feat_cnt, targets=targets)
        elif processing in {PreprocessingMethod.CATBOOST, PreprocessingMethod.CATBOOST_OPT}:
            return cls.for_catboost(raw, splits=splits, feat_cnt=feat_cnt, device=device, processing=processing)
        elif processing == PreprocessingMethod.TREES:
            return cls.for_trees_mlp(raw, splits=splits, targets=targets, feat_cnt=feat_cnt, device=device)
        elif processing == PreprocessingMethod.TREES_OPT:
            return cls.for_trees_opt(raw, splits=splits, targets=targets, feat_cnt=feat_cnt, device=device)
        else:
            raise ValueError(f"Unsupported processing method: {processing}")

    @classmethod
    def for_tabstar(cls, raw: RawDataset, splits: List[DataSplit], targets: List[str], feat_cnt: Dict,
                    number_verbalization: NumberVerbalization) -> Self:
        assert isinstance(number_verbalization, NumberVerbalization)
        x_txt, x_num = scale_x_num_and_add_categorical_bins(raw=raw, splits=splits,
                                                            number_verbalization=number_verbalization)
        verbalize_x_txt(x_txt)
        properties = DatasetProperties.create(raw=raw, splits=splits, feat_cnt=feat_cnt, targets=targets,
                                              processing=PreprocessingMethod.TABSTAR)
        x_txt, x_num = add_target_tokens(x_txt=x_txt, x_num=x_num, data=properties)
        dataset = TabularDataset(properties=properties, x=x_txt, y=raw.y, splits=splits, x_num=x_num)
        return dataset


    @classmethod
    def for_trees_mlp(cls, raw: RawDataset, splits: List[DataSplit], targets: List[str], feat_cnt: Dict,
                      device: torch.device) -> Self:
        fill_median_for_numerical_nulls(raw=raw, splits=splits)
        transform_texts_to_embeddings(raw=raw, device=device)
        process_x_cat_to_indices(raw=raw, splits=splits)
        dataset = cls.from_processed(raw=raw, processing=PreprocessingMethod.TREES, splits=splits, feat_cnt=feat_cnt,
                                     targets=targets)
        return dataset

    @classmethod
    def for_trees_opt(cls, raw: RawDataset, splits: List[DataSplit], targets: List[str], feat_cnt: Dict,
                      device: torch.device) -> Self:
        # We avoid filling the median and doing categorical encoding, as we'll do it per-split
        transform_texts_to_embeddings(raw=raw, device=device)
        dataset = cls.from_processed(raw=raw, processing=PreprocessingMethod.TREES_OPT, splits=splits,
                                     feat_cnt=feat_cnt, targets=targets)
        return dataset

    @classmethod
    def for_catboost(cls, raw: RawDataset, splits: List[DataSplit], feat_cnt: Dict,
                     device: torch.device, processing: PreprocessingMethod) -> Self:
        transform_texts_to_embeddings(raw=raw, device=device)
        dataset = cls.from_processed(raw=raw, processing=processing, splits=splits, feat_cnt=feat_cnt)
        return dataset

    @classmethod
    def from_processed(cls, raw: RawDataset, processing: PreprocessingMethod, splits: List[DataSplit], feat_cnt: Dict,
                       targets: Optional[List[str]] = None) -> Self:
        properties = DatasetProperties.create(raw, splits=splits, feat_cnt=feat_cnt, processing=processing,
                                              targets=targets)
        dataset = TabularDataset(properties=properties, x=raw.x, y=raw.y, splits=splits)
        return dataset
