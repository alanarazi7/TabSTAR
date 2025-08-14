from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel



class LightGBM(TabularModel):

    MODEL_NAME = "LightGBM 💡"
    SHORT_NAME = "light"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> LGBMRegressor | LGBMClassifier:
        model_cls = LGBMClassifier if self.is_cls else LGBMRegressor
        model = model_cls(verbose=-1, random_state=SEED)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=self.categorical_indices)
