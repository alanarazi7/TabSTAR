from pandas import DataFrame, Series
from tabicl import TabICLClassifier

from tabstar_paper.baselines.abstract_model import TabularModel



class TabICL(TabularModel):

    MODEL_NAME = "TabICL 🤖"
    SHORT_NAME = "icl"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> TabICLClassifier:
        if not self.is_cls:
            raise ValueError("TabICL is only supported for classification tasks for now.")
        model = TabICLClassifier(device=str(self.device), checkpoint_version="tabicl-classifier-v1-0208.ckpt")
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)

