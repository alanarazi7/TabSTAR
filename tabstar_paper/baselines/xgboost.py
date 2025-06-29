from dataclasses import dataclass, asdict
from typing import Tuple

from xgboost import XGBRegressor, XGBClassifier
from pandas import DataFrame, Series

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features
from tabstar_paper.baselines.preprocessing.numerical import fill_median
from tabstar_paper.baselines.preprocessing.categorical import CategoricalEncoder
from tabstar_paper.utils.logging import log_all_methods


@dataclass
class XGBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    booster: str = "gbtree"

@log_all_methods
class XGBoost(TabularModel):

    MODEL_NAME = "XGBoost 🌲"
    SHORT_NAME = "xgb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical_features = []

    def initialize_model(self) -> XGBRegressor | XGBClassifier:
        model_cls = XGBClassifier if self.is_cls else XGBRegressor
        params = XGBoostDefaultHyperparams()
        model = model_cls(**asdict(params))
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        # TODO: we should initialize these objects in the constructor
        self.numerical_fillers = {col: fill_median(x[col]) for col in self.numerical_features}
        self.categorical_encoders = {}
        for col in self.categorical_features:
            enc = CategoricalEncoder()
            enc.fit(x[col])
            self.categorical_encoders[col] = enc
        self.text_transformers = fit_text_encoders(x=x, text_features=self.text_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        for col, filler in self.numerical_fillers.items():
            x[col] = x[col].fillna(filler.median)
        for col in self.categorical_features:
            x[col] = self.categorical_encoders[col].transform(x[col])
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=self.verbose)