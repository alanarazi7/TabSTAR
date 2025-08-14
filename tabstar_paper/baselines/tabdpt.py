from pandas import DataFrame, Series
from tabdpt import TabDPTRegressor, TabDPTClassifier

from tabstar_paper.baselines.abstract_model import TabularModel


class TabDPT(TabularModel):

    MODEL_NAME = "TabDPT 6️⃣"
    SHORT_NAME = "dpt"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> TabDPTClassifier | TabDPTRegressor:
        model_cls = TabDPTClassifier if self.is_cls else TabDPTRegressor
        model = model_cls(device=str(self.device), use_flash=False)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        self.model_.fit(x_train, y_train)

