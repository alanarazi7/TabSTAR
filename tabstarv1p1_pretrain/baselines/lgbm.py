from typing import Optional

from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series

from tabstarv1p1_pretrain.constants import SEED
from tabstarv1p1_pretrain.baselines.abstract_model import TabularModel


class LightGBM(TabularModel):

    MODEL_NAME = "LightGBM 💡"
    SHORT_NAME = "light"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def __init__(self, *args, num_threads: Optional[int] = None, **kwargs):
        # LightGBM's own thread auto-detection reads hardware core count directly and calls
        # omp_set_num_threads() with it, overriding OMP_NUM_THREADS rather than respecting it.
        # Callers fanning out many fits across worker processes (e.g. t4_data.py) pass this
        # to cap threads per fit; left unset, default (use-all-cores) is unaffected.
        self.num_threads = num_threads
        super().__init__(*args, **kwargs)

    def initialize_model(self) -> LGBMRegressor | LGBMClassifier:
        model_cls = LGBMClassifier if self.is_cls else LGBMRegressor
        params = {"verbose": -1, "random_state": SEED}
        if self.num_threads is not None:
            params["num_threads"] = self.num_threads
        return model_cls(**params)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=self.categorical_indices)
