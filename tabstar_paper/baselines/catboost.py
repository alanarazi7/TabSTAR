from dataclasses import dataclass, asdict
from typing import Tuple

from catboost import CatBoostRegressor, CatBoostClassifier
from pandas import DataFrame, Series

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders


@dataclass
class CatBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    early_stopping_rounds: int = 50
    iterations: int = 2000
    od_pval: float = 0.001


class CatBoost(TabularModel):

    MODEL_NAME = "CatBoost 😸"
    SHORT_NAME = "cat"

    def initialize_model(self):
        model_cls = CatBoostClassifier if self.is_cls else CatBoostRegressor
        params = CatBoostDefaultHyperparams()
        self.model_ = model_cls(**asdict(params))

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.text_transformers = fit_text_encoders(x=x, numerical_features=self.numerical_features, device=self.device)

    def transform_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        raise NotImplementedError("Transform preprocessor method not implemented yet")

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        non_cat_features = set(self.numerical_features).union(set(self.text_transformers))
        cat_features = [i for i, c in enumerate(x_train.columns) if c in non_cat_features]
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features)

