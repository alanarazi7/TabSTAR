from dataclasses import dataclass, asdict
from typing import Tuple

from xgboost import XGBRegressor, XGBClassifier
from pandas import DataFrame, Series

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features

@dataclass
class XGBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    booster: str = "gbtree"
    verbosity: int = 1

class XGBoost(TabularModel):

    MODEL_NAME = "XGBoost 🌲"
    SHORT_NAME = "xgb"

    def initialize_model(self) -> XGBRegressor | XGBClassifier:
        model_cls = XGBClassifier if self.is_cls else XGBRegressor
        params = XGBoostDefaultHyperparams()
        model = model_cls(**asdict(params))
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.text_transformers = fit_text_encoders(x=x, numerical_features=self.numerical_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        numerical_features = set(self.numerical_features)
        text_features = {f"{c}_{str(n+1).zfill(2)}" for c, e in self.text_transformers.items() for n in range(e.n_components)}
        assert all(c in x_train.columns for c in text_features)
        non_cat_features = numerical_features.union(text_features)
        # For XGBoost, categorical features are handled as numerics or via encoding
        eval_set = [(x_val, y_val)]
        fit_args = dict(eval_set=eval_set, early_stopping_rounds=50, verbose=self.verbose)
        self.model_.fit(x_train, y_train, **fit_args)
        self.model_.fit(x_train, y_train, **fit_args)
