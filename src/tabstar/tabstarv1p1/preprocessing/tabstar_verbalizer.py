from typing import Optional, Dict, List, Union

import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skrub import DatetimeEncoder

from tabstar.tabstarv1p1.preprocessing.feat_types import detect_numerical_features, transform_feature_types
from tabstar.tabstarv1p1.preprocessing.nulls import raise_if_null_target
from tabstar.tabstarv1p1.preprocessing.sparse import densify_objects
from tabstar.tabstarv1p1.preprocessing.target import fit_cls_y, transform_cls_y
from tabstar.tabstarv1p1.preprocessing.texts import replace_column_names, replace_whitespaces
from tabstar.tabstarv1p1.preprocessing.verbalize import prepend_target_tokens, verbalize_textual_features

from tabstar.tabstarv1p1.preprocessing.constant_columns import detect_constant_columns, drop_constant_columns
from tabstar.tabstarv1p1.preprocessing.dates import fit_date_encoders, transform_date_features
from tabstar.tabstarv1p1.preprocessing.feature_selection import FeatureSelector, MAX_FEATURES
from tabstar.tabstarv1p1.preprocessing.numerical import RichNumericalTransformer
from tabstar.tabstarv1p1.constants import REG_Z_CLIP
from tabstar.tabstarv1p1.preprocessing.verbalizer import TabSTARData


class TabSTARv1p1Verbalizer:
    def __init__(self,
                 is_cls: bool,
                 d_num: int,
                 verbose: bool = False,
                 max_features: int = MAX_FEATURES):
        self.is_cls = is_cls
        self.d_num = d_num
        self.max_features = max_features
        self.verbose = verbose
        self.date_transformers: Dict[str, DatetimeEncoder] = {}
        self.numerical_transformers: Dict[str, RichNumericalTransformer] = {}
        self.target_transformer: Optional[Union[LabelEncoder, StandardScaler]] = None
        self.d_output: Optional[int] = None
        self.y_name: Optional[str] = None
        self.y_values: Optional[List[str]] = None
        self.constant_columns: List[str] = []
        self.feature_selector: Optional[FeatureSelector] = None

    def fit(self, X, y):
        x = X.copy()
        y = y.copy()
        raise_if_null_target(y)
        self.assert_no_duplicate_columns(x)
        x, y = densify_objects(x=x, y=y)
        self.date_transformers = fit_date_encoders(x=x)
        self.vprint(f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}")
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x, y = replace_column_names(x=x, y=y)
        numerical_features = detect_numerical_features(x)
        self.vprint(f"🔢 Detected {len(numerical_features)} numerical features: {sorted(numerical_features)}")
        text_features = [col for col in x.columns if col not in numerical_features]
        self.vprint(f"📝 Detected {len(text_features)} textual features: {sorted(text_features)}")
        x = transform_feature_types(x=x, numerical_features=numerical_features)
        self.constant_columns = detect_constant_columns(x)
        self.vprint(f"🧊 Detected {len(self.constant_columns)} constant features: {sorted(self.constant_columns)}")
        x = drop_constant_columns(x=x, constant_columns=self.constant_columns)
        numerical_features = {col for col in numerical_features if col in x.columns}
        if self.is_cls:
            self.target_transformer = fit_cls_y(y)
        else:
            self.target_transformer = StandardScaler().fit(y.values.reshape(-1, 1))
        if self.is_cls:
            self.d_output = len(self.target_transformer.classes_)
        else:
            self.d_output = 1
        self.feature_selector = FeatureSelector(max_features=self.max_features)
        self.feature_selector.fit(x=x, y=y, is_cls=self.is_cls, numerical_features=numerical_features)
        x = self.feature_selector.transform(x)
        numerical_features = self.feature_selector.numerical_columns
        for col in numerical_features:
            self.numerical_transformers[col] = RichNumericalTransformer.fit_series(s=x[col], d_num=self.d_num)
        self.y_name = str(y.name)
        if self.is_cls:
            self.y_values = sorted(self.target_transformer.classes_)

    def transform(self, x: DataFrame, y: Optional[Series]) -> TabSTARData:
        x = x.copy()
        if y is not None:
            y = y.copy()
        self.assert_no_duplicate_columns(x)
        x, y = densify_objects(x=x, y=y)
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x, y = replace_column_names(x=x, y=y)
        x = drop_constant_columns(x=x, constant_columns=self.constant_columns)
        x = self.feature_selector.transform(x)
        num_cols = sorted(self.numerical_transformers)
        x = transform_feature_types(x=x, numerical_features=set(num_cols))
        y = self.transform_target(y=y)
        x = verbalize_textual_features(x=x)
        x = prepend_target_tokens(x=x, y_name=self.y_name, y_values=self.y_values)
        text_cols = [col for col in x.columns if col not in num_cols]
        x_txt = x[text_cols + num_cols].copy()
        n_samples, seq_len = x_txt.shape
        # x_num will hold the numerical features, and zero otherwise
        x_num = np.zeros((n_samples, seq_len, self.d_num), dtype=np.float32)
        for col in num_cols:
            i = x_txt.columns.get_loc(col)
            x_num[:, i, :] = self.numerical_transformers[col].transform(x[col])
            x_txt[col] = f"Numerical Feature: {replace_whitespaces(col)}"
        x_txt = x_txt.to_numpy()
        data = TabSTARData(d_output=self.d_output, x_txt=x_txt, x_num=x_num, y=y)
        return data

    def transform_target(self, y: Optional[Series]) -> Optional[Union[Series, np.ndarray]]:
        if y is None:
            return None
        y = y.copy()
        raise_if_null_target(y)
        if self.is_cls:
            return transform_cls_y(y=y, encoder=self.target_transformer)
        return self._transform_reg_target(y)

    def _transform_reg_target(self, y: Series) -> Series:
        # tabstar.preprocessing.target.transform_preprocess_y clips to +/-3 (Z_MAX_ABS_VAL,
        # hardcoded in the pinned tabstar package); reimplemented here to clip to REG_Z_CLIP instead.
        assert isinstance(self.target_transformer, StandardScaler)
        z = self.target_transformer.transform(y.values.reshape(-1, 1)).flatten()
        z = z.clip(-REG_Z_CLIP, REG_Z_CLIP)
        return Series(z, index=y.index, name=y.name)

    def inverse_transform_target(self, y):
        y = y.copy()
        assert isinstance(self.target_transformer, StandardScaler)
        return self.target_transformer.inverse_transform(y)

    @staticmethod
    def assert_no_duplicate_columns(x: DataFrame):
        if len(set(x.columns)) != len(x.columns):
            raise ValueError("Duplicate column names found in DataFrame!")

    def vprint(self, s: str):
        if self.verbose:
            print(s)
