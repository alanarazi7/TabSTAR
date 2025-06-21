from dataclasses import dataclass
from typing import Optional, Dict, List, Union

import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from skrub import DatetimeEncoder

from tabstar.preprocessing.binning import fit_numerical_bins, transform_numerical_bins
from tabstar.preprocessing.dates import fit_date_encoders, transform_date_features
from tabstar.preprocessing.feat_types import detect_numerical_features, transform_feature_types
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.scaler import fit_standard_scaler, transform_clipped_z_scores
from tabstar.preprocessing.sparse import densify_objects
from tabstar.preprocessing.target import fit_preprocess_y
from tabstar.preprocessing.texts import replace_column_names
from tabstar.preprocessing.verbalize import prepend_target_tokens, verbalize_textual_features


@dataclass
class TabSTARData:
    d_output: int
    x_txt: Union[DataFrame, np.ndarray]
    x_num: np.ndarray
    y: Optional[Series] = None

    def __len__(self) -> int:
        return len(self.x_txt)

class TabSTARVerbalizer:
    def __init__(self, is_cls: bool, verbose: bool = False):
        self.is_cls = is_cls
        self.verbose = verbose
        self.date_transformers: Dict[str, DatetimeEncoder] = {}
        self.numerical_transformers: Dict[str, StandardScaler] = {}
        self.semantic_transformers: Dict[str, QuantileTransformer] = {}
        self.target_transformer: Optional[Union[LabelEncoder, StandardScaler]] = None
        self.d_output: Optional[int] = None
        self.y_name: Optional[str] = None
        self.y_values: Optional[List[str]] = None
        self.constant_columns: List[str] = []

    def fit(self, X, y):
        if len(X.columns) >= 200:
            print("⚠️ Warning: More than 200 columns detected. This will probably lead to memory issues.")
        raise_if_null_target(y)
        self.assert_no_duplicate_columns(X)
        x, y = densify_objects(x=X, y=y)
        self.date_transformers = fit_date_encoders(x=x)
        self.vprint(f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}")
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x, y = replace_column_names(x=x, y=y)
        numerical_features = detect_numerical_features(x)
        self.vprint(f"🔢 Detected {len(numerical_features)} numerical features: {sorted(numerical_features)}")
        text_features = [col for col in x.columns if col not in numerical_features]
        self.vprint(f"📝 Detected {len(text_features)} textual features: {sorted(text_features)}")
        x = transform_feature_types(x=x, numerical_features=numerical_features)
        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        if self.is_cls:
            self.d_output = len(self.target_transformer.classes_)
        else:
            self.d_output = 1
        self.constant_columns = [col for col in x.columns if x[col].nunique() == 1]
        for col in numerical_features:
            if col in self.constant_columns:
                continue
            self.numerical_transformers[col] = fit_standard_scaler(s=x[col])
            self.semantic_transformers[col] = fit_numerical_bins(s=x[col])
        self.y_name = str(y.name)
        if self.is_cls:
            self.y_values = sorted(self.target_transformer.classes_)

    def transform(self, x: DataFrame, y: Optional[Series]) -> TabSTARData:
        self.assert_no_duplicate_columns(x)
        x, y = densify_objects(x=x, y=y)
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x, y = replace_column_names(x=x, y=y)
        num_cols = sorted(self.numerical_transformers)
        x = transform_feature_types(x=x, numerical_features=set(num_cols))
        y = self.transform_target(y=y)
        x = verbalize_textual_features(x=x)
        x = x.drop(columns=self.constant_columns, errors='ignore')
        x = prepend_target_tokens(x=x, y_name=self.y_name, y_values=self.y_values)
        text_cols = [col for col in x.columns if col not in num_cols]
        x_txt = x[text_cols + num_cols].copy()
        # x_num will hold the numerical features transformed to z-scores, and zero otherwise
        x_num = np.zeros(shape=x.shape, dtype=np.float32)
        for col in num_cols:
            x_txt[col] = transform_numerical_bins(s=x[col], scaler=self.semantic_transformers[col])
            idx = x_txt.columns.get_loc(col)
            s_num = transform_clipped_z_scores(s=x[col], scaler=self.numerical_transformers[col], allow_null=True)
            x_num[:, idx] = s_num.to_numpy()
        x_txt = x_txt.to_numpy()
        data = TabSTARData(d_output=self.d_output, x_txt=x_txt, x_num=x_num, y=y)
        return data

    def transform_target(self, y: Optional[Series]) -> Optional[np.ndarray]:
        if y is None:
            return None
        raise_if_null_target(y)
        if isinstance(self.target_transformer, LabelEncoder):
            return self.target_transformer.transform(y)
        elif isinstance(self.target_transformer, StandardScaler):
            return self.target_transformer.transform(y.values.reshape(-1, 1)).flatten()
        else:
            raise TypeError("Unsupported target transformer type.")

    def inverse_transform_target(self, y):
        assert isinstance(self.target_transformer, StandardScaler)
        return self.target_transformer.inverse_transform(y)

    @staticmethod
    def assert_no_duplicate_columns(x: DataFrame):
        if len(set(x.columns)) != len(x.columns):
            raise ValueError("Duplicate column names found in DataFrame!")

    def vprint(self, s: str):
        if self.verbose:
            print(s)
