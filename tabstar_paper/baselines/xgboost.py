from dataclasses import dataclass, asdict
from typing import Tuple

from xgboost import XGBRegressor, XGBClassifier
from pandas import DataFrame, Series

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features
from tabstar_paper.baselines.preprocessing.numerical import fill_median
from tabstar_paper.baselines.preprocessing.categorical import fill_mode, CategoricalEncoder
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
        # Detect categorical features
        # TODO: we should use string detector function, not hardcoded "object"
        self.categorical_features = [
            col for col in x.columns
            if col not in self.numerical_features
            and x[col].dtype == "object"
        ]
        # Numerical
        # TODO: we should initialize these objects in the constructor
        self.numerical_fillers = {col: fill_median(x[col]) for col in self.numerical_features}
        # Categorical
        self.categorical_fillers = {col: fill_mode(x[col]) for col in self.categorical_features}
        self.categorical_encoders = {}
        for col, filler in self.categorical_fillers.items():
            enc = CategoricalEncoder()
            enc.fit(filler.src)
            self.categorical_encoders[col] = enc
        # Text
        self.text_transformers = fit_text_encoders(x=x, numerical_features=self.numerical_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        # TODO: (1) don't use `getattr` here, super ugly and not relevant
        # TODO: (2) The comments are not needed, it's clear from the code what is being done
        # Numerical
        for col, filler in getattr(self, "numerical_fillers", {}).items():
            if col in x:
                x[col] = x[col].fillna(filler.median)
        # Categorical
        for col, filler in getattr(self, "categorical_fillers", {}).items():
            if col in x:
                x[col] = x[col].fillna(filler.mode)
                x[col] = self.categorical_encoders[col].transform(x[col])
        # Text
        # TODO: You must put this back, it is essential for the model to work with text features
        # x = transform_text_features(x=x, text_encoders=getattr(self, "text_transformers", {})) i removed this line and it worked
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=self.verbose)