from dataclasses import dataclass, asdict
from typing import Tuple

from catboost import CatBoostRegressor, CatBoostClassifier
from pandas import DataFrame, Series

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features


@dataclass
class CatBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    early_stopping_rounds: int = 50
    iterations: int = 2000
    od_pval: float = 0.001


class CatBoost(TabularModel):

    MODEL_NAME = "CatBoost 😸"
    SHORT_NAME = "cat"

    def initialize_model(self) -> CatBoostRegressor | CatBoostClassifier:
        model_cls = CatBoostClassifier if self.is_cls else CatBoostRegressor
        params = CatBoostDefaultHyperparams()
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
        cat_features = [i for i, c in enumerate(x_train.columns) if c not in non_cat_features]
        logging_level = "Verbose" if self.verbose else "Silent"
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features,
                        logging_level=logging_level)

