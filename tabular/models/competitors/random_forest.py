from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tabular.evaluation.sklearn_model import init_model
from tabular.models.abstract_sklearn import TabularSklearnModel
from tabular.preprocessing.objects import PreprocessingMethod

@dataclass
class RandomForestHyperparams:
    n_estimators: int = 100



class RandomForest(TabularSklearnModel):

    MODEL_NAME = "RandomForest 🌳"
    SHORT_NAME = "rf"
    PROCESSING = PreprocessingMethod.TREES

    def initialize_model(self):
        self.model = init_model(config=self.config, is_reg=self.dataset.is_regression,
                                classifier_cls=RandomForestClassifier, regressor_cls=RandomForestRegressor)

    def train(self):
        x_train, y_train, x_dev, y_dev = self.load_all()
        self.model.fit(x_train, y_train)

    def set_config(self) -> RandomForestHyperparams:
        return RandomForestHyperparams()