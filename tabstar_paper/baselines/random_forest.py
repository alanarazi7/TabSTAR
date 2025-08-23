from dataclasses import dataclass, asdict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel

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
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> RandomForestRegressor | RandomForestClassifier:
        model_cls = RandomForestClassifier if self.is_cls else RandomForestRegressor
        params = RandomForestDefaultHyperparams()
        model = model_cls(**vars(params))
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)