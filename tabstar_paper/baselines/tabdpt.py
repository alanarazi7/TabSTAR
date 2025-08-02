from typing import Tuple

from pandas import DataFrame, Series
from tabdpt import TabDPTRegressor, TabDPTClassifier

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.categorical import fit_categorical_encoders, transform_categorical_features
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features



class TabDPT(TabularModel):

    MODEL_NAME = "TabDPT 6️⃣"
    SHORT_NAME = "dpt"
    USE_VAL_SPLIT = False

    def initialize_model(self) -> TabDPTClassifier | TabDPTRegressor:
        model_cls = TabDPTClassifier if self.is_cls else TabDPTRegressor
        model = model_cls(device=str(self.device), use_flash=False)
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.categorical_encoders = fit_categorical_encoders(x=x, categorical_features=self.categorical_features)
        self.text_transformers = fit_text_encoders(x=x, text_features=self.text_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        x = transform_categorical_features(x=x, categorical_encoders=self.categorical_encoders)
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        self.model_.fit(x_train, y_train)

