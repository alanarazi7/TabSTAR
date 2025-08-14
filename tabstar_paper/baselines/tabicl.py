from typing import Tuple

from pandas import DataFrame, Series
from tabicl import TabICLClassifier

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features



class TabICL(TabularModel):

    MODEL_NAME = "TabICL 🤖"
    SHORT_NAME = "icl"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False

    def initialize_model(self) -> TabICLClassifier:
        if not self.is_cls:
            raise ValueError("TabICL is only supported for classification tasks for now.")
        model = TabICLClassifier(device=str(self.device), checkpoint_version="tabicl-classifier-v1-0208.ckpt")
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.text_transformers = fit_text_encoders(x=x, text_features=self.text_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)

