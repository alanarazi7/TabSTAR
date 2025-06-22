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


class XGBoost(TabularModel):

    MODEL_NAME = "XGBoost 🌲"
    SHORT_NAME = "xgb"

    def initialize_model(self) -> XGBRegressor | XGBClassifier:
        model_cls = XGBClassifier if self.is_cls else XGBRegressor
        params = XGBoostDefaultHyperparams()
        model = model_cls(**asdict(params))
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        # todo might need numerical and categorical features preprocessing from:
        # https://github.com/alanarazi7/TabSTAR/blob/master/tabular/datasets/data_processing.py#L68
        self.text_transformers = fit_text_encoders(x=x, numerical_features=self.numerical_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        # same preprocessing might be needed here as well
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=self.verbose)