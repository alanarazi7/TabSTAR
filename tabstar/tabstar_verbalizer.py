from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder

from tabstar.preprocessing.binning import fit_numerical_bins, transform_numerical_bins
from tabstar.preprocessing.scaler import fit_standard_scaler, transform_clipped_z_scores
from tabstar.preprocessing.target import fit_preprocess_y, transform_preprocess_y
from tabstar.preprocessing.verbalize import prepend_target_tokens


@dataclass
class TabSTARData:
    d_output: int
    x_txt: DataFrame | np.ndarray
    x_num: np.ndarray
    y: Optional[Series] = None

class TabSTARVerbalizer:
    def __init__(self, is_cls: bool):
        self.is_cls = is_cls
        self.numerical_transformers: Dict[str, StandardScaler] = {}
        self.semantic_transformers: Dict[str, QuantileTransformer] = {}
        self.target_transformer: Optional[LabelEncoder | StandardScaler] = None
        self.d_output: Optional[int] = None
        self.y_name: Optional[str] = None
        self.y_values: Optional[List[str]] = None

    def fit(self, X, y):
        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        if self.is_cls:
            self.d_output = len(self.target_transformer.classes_)
        else:
            self.d_output = 1
        numeric_cols = [col for col, dtype in X.dtypes.items() if is_numeric_dtype(dtype)]
        for col in numeric_cols:
            self.numerical_transformers[col] = fit_standard_scaler(s=X[col])
            self.semantic_transformers[col] = fit_numerical_bins(s=X[col])
        self.y_name = str(y.name)
        if self.is_cls:
            self.y_values = sorted(self.target_transformer.classes_)

    def transform(self, x: DataFrame, y: Optional[Series]) -> TabSTARData:
        if y is not None:
            y = transform_preprocess_y(y=y, scaler=self.target_transformer)
        x = prepend_target_tokens(x=x, y_name=self.y_name, y_values=self.y_values)
        num_cols = list(set(self.numerical_transformers))
        text_cols = [col for col in x.columns if col not in num_cols]
        x_txt = x[text_cols + num_cols].copy()
        # x_num will hold the numerical features transformed to z-scores, and zero otherwise
        x_num = np.zeros(shape=x.shape, dtype=np.float32)
        for col in num_cols:
            x_txt[col] = transform_numerical_bins(s=x[col], scaler=self.semantic_transformers[col])
            idx = x_txt.columns.get_loc(col)
            s_num = transform_clipped_z_scores(s=x[col], scaler=self.numerical_transformers[col])
            x_num[:, idx] = s_num.to_numpy()
        x_txt = x_txt.to_numpy()
        data = TabSTARData(d_output=self.d_output, x_txt=x_txt, x_num=x_num, y=y)
        return data

    def inverse_transform_target(self, y):
        # if self.target_transformer_ is not None:
        #     return self.target_transformer_.inverse_transform(y)
        # return y
        raise NotImplementedError
