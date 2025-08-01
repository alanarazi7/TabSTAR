from typing import Tuple

from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel
from tabpfn_client import TabPFNClassifier, TabPFNRegressor, set_access_token

from tabstar_paper.constants import TABPFN_TOKEN

MAX_SAMPLES = 10_000

class TabPFNv2(TabularModel):

    MODEL_NAME = "TabPFN-v2 🤯"
    SHORT_NAME = "pfn"
    DO_VAL_SPLIT = False

    def initialize_model(self) -> TabPFNClassifier | TabPFNRegressor:
        # TODO: Move away from closed-source client version, as this isn't reproducible, they improve the model
        assert TABPFN_TOKEN, f"Please set the TABPFN_TOKEN environment variable to use TabPFN."
        set_access_token(TABPFN_TOKEN)
        model_cls = TabPFNRegressor if not self.is_cls else TabPFNClassifier
        model = model_cls(random_state=SEED)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        if len(x_train) > MAX_SAMPLES:
            raise RuntimeError(f"TabPFN is not designed to handle datasets larger than {MAX_SAMPLES} samples. ")
        self.model_.fit(x_train, y_train)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        pass

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        return x, y
