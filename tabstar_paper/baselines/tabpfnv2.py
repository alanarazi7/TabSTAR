from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel
from tabpfn import TabPFNClassifier, TabPFNRegressor


MAX_SAMPLES = 10_000
MAX_FEATURES = 500

class TabPFNv2(TabularModel):

    MODEL_NAME = "TabPFNv2 🧪"
    SHORT_NAME = "tabpfn"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        if len(x_train) > MAX_SAMPLES:
            raise RuntimeError(f"TabPFN is not designed to handle datasets larger than {MAX_SAMPLES} samples. ")
        if len(x_train.columns) > MAX_FEATURES:
            print(f"🚨 Warning: {len(x_train.columns)} features detected, ignoring pretraining limits.")
        cat_feature = [i for i, c in enumerate(x_train.columns) if c in self.categorical_features]
        model_cls = TabPFNRegressor if not self.is_cls else TabPFNClassifier
        self.model_ = model_cls(random_state=SEED, categorical_features_indices=cat_feature,
                                ignore_pretraining_limits=True)
        self.model_.fit(x_train, y_train)
