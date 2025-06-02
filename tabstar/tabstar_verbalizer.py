from typing import Optional, Dict, Tuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder

from tabstar.preprocessing.binning import fit_numerical_bins, transform_numerical_bins
from tabstar.preprocessing.scaler import fit_standard_scaler, transform_clipped_z_scores
from tabstar.preprocessing.target import fit_preprocess_y, transform_preprocess_y


class TabSTARVerbalizer:
    def __init__(self, is_cls: bool):
        self.is_cls = is_cls
        self.numerical_transformers: Dict[str, StandardScaler] = {}
        self.semantic_transformers: Dict[str, QuantileTransformer] = {}
        self.target_transformer: Optional[LabelEncoder | StandardScaler] = None

    def fit(self, X, y):
        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        for col, dtype in X.dtypes.items():
            if dtype == 'str':
                continue
            assert dtype == 'float', f"Column {col} has unsupported dtype {dtype}. Expected float or str."
            self.numerical_transformers[col] = fit_standard_scaler(s=X[col])
            self.semantic_transformers[col] = fit_numerical_bins(s=X[col])

    def transform(self, x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, np.ndarray, Optional[Series]]:
        if y is not None:
            y = transform_preprocess_y(y=y, scaler=self.target_transformer)
        num_cols = list(set(self.numerical_transformers))
        text_cols = [col for col in x.columns if col not in num_cols]
        x_txt = x[text_cols + num_cols].copy()
        x_num = np.zeros(shape=x.shape, dtype=np.float32)
        for col in num_cols:
            x_txt[col] = transform_clipped_z_scores(s=x[col], scaler=self.numerical_transformers[col])
            s_sem = transform_numerical_bins(s=x[col], scaler=self.semantic_transformers[col])
            # Align x_num with x_txt: only fill semantic bins in the positions of numerical columns; keep others zero.
            idx = x_txt.columns.get_loc(col)
            x_num[:, idx] = s_sem.to_numpy()
        return x_txt, x_num, y

    def inverse_transform_target(self, y):
        # if self.target_transformer_ is not None:
        #     return self.target_transformer_.inverse_transform(y)
        # return y
        raise NotImplementedError
