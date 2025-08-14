from dataclasses import dataclass, asdict
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features


@dataclass
class RandomForestDefaultHyperparams:
    n_estimators: int = 100
    random_state: int = SEED


class RandomForest(TabularModel):

    MODEL_NAME = "RandomForest 🌳"
    SHORT_NAME = "rf"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True

    def initialize_model(self) -> RandomForestRegressor | RandomForestClassifier:
        model_cls = RandomForestClassifier if self.is_cls else RandomForestRegressor
        params = RandomForestDefaultHyperparams()
        model = model_cls(**asdict(params))
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.text_transformers = fit_text_encoders(x=x, text_features=self.text_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)