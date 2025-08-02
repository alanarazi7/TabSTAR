from typing import Tuple, Dict, Optional, Set

import numpy as np
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skrub import DatetimeEncoder, TextEncoder

from tabstar.preprocessing.dates import fit_date_encoders, transform_date_features
from tabstar.preprocessing.feat_types import detect_numerical_features, transform_feature_types
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.sparse import densify_objects
from tabstar.preprocessing.splits import split_to_val
from tabstar.preprocessing.target import fit_preprocess_y, transform_preprocess_y
from tabstar.training.metrics import calculate_metric, Metrics
from tabstar_paper.baselines.preprocessing.feat_types import classify_semantic_features
from tabstar_paper.constants import CPU
from tabstar_paper.datasets.objects import SupervisedTask


class TabularModel:

    MODEL_NAME: str
    SHORT_NAME: str
    USE_VAL_SPLIT: bool

    def __init__(self, problem_type: SupervisedTask, device: torch.device, verbose: bool = False):
        assert problem_type in {SupervisedTask.REGRESSION, SupervisedTask.BINARY, SupervisedTask.MULTICLASS}
        self.problem_type = problem_type
        self.is_cls = bool(problem_type in {SupervisedTask.BINARY, SupervisedTask.MULTICLASS})
        self.device = device
        if CPU:
            self.device = torch.device("cpu")
        self.verbose = verbose
        self.model_ = self.initialize_model()
        self.d_output: int = 0
        self.target_transformer: Optional[LabelEncoder | StandardScaler] = None
        self.date_transformers: Dict[str, DatetimeEncoder] = {}
        self.numerical_features: Set[str] = set()
        self.numerical_medians: Dict[str, float] = {}
        self.categorical_features: Set[str] = set()
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.text_features: Set[str] = set()
        self.text_transformers: Dict[str, TextEncoder] = {}

    def initialize_model(self):
        raise NotImplementedError("Initialize model method not implemented yet")

    def fit(self, x: DataFrame, y: Series):
        x_train, y_train = x.copy(), y.copy()
        if self.USE_VAL_SPLIT:
            x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        else:
            x_val, y_val = None, None
        self.fit_preprocessor(x_train=x_train, y_train=y_train)
        x_train, y_train = self.transform_preprocessor(x=x_train, y=y_train)
        if x_val is not None and y_val is not None:
            x_val, y_val = self.transform_preprocessor(x=x_val, y=y_val)
        self.fit_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    def fit_preprocessor(self, x_train: DataFrame, y_train: Series):
        x_train, y_train = self.do_model_agnostic_preprocessing(x=x_train, y=y_train)
        return self.fit_internal_preprocessor(x=x_train, y=y_train)

    def transform_preprocessor(self, x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
        x, y = densify_objects(x=x, y=y)
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x = transform_feature_types(x=x, numerical_features=self.numerical_features)
        if y is not None:
            raise_if_null_target(y)
            y = transform_preprocess_y(y=y, scaler=self.target_transformer)
        return self.transform_internal_preprocessor(x=x, y=y)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        raise NotImplementedError("Fit internal preprocessor method not implemented yet")

    def transform_internal_preprocessor(self, x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
        raise NotImplementedError("Transform internal preprocessor method not implemented yet")

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: Optional[DataFrame], y_val: Optional[Series]):
        raise NotImplementedError("Fit model method not implemented yet")

    def do_model_agnostic_preprocessing(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        raise_if_null_target(y)
        x, y = densify_objects(x=x, y=y)
        self.date_transformers = fit_date_encoders(x=x)
        self.vprint(f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}")
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        self.numerical_features = detect_numerical_features(x)
        self.vprint(f"🔢 Detected {len(self.numerical_features)} numerical features: {sorted(self.numerical_features)}")
        x = transform_feature_types(x=x, numerical_features=self.numerical_features)
        semantic_types = classify_semantic_features(x=x, numerical_features=self.numerical_features)
        self.categorical_features = semantic_types.categorical_features
        self.text_features = semantic_types.text_features
        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        if self.is_cls:
            self.d_output = len(self.target_transformer.classes_)
        else:
            self.d_output = 1
        # TODO: drop constant columns, where constants means all values are the same (and no nulls)
        return x, y

    def predict(self, x: DataFrame) -> np.ndarray:
        x, _ = self.transform_preprocessor(x=x, y=None)
        if not self.is_cls:
            return self.model_.predict(x)
        probs = self.model_.predict_proba(x)
        if self.d_output == 2:
            probs = probs[:, 1]
        return probs

    def score(self, X, y) -> float:
        metrics = self.score_all_metrics(X=X, y=y)
        return metrics.score

    def score_all_metrics(self, X, y) -> Metrics:
        x = X.copy()
        y = y.copy()
        y_true = transform_preprocess_y(y=y, scaler=self.target_transformer)
        y_pred = self.predict(x)
        metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=self.d_output)
        return metrics

    def vprint(self, s: str):
        if self.verbose:
            print(s)
