from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from tabstar.arch.arch import TabStarModel
from tabstar.preprocessing.preprocess import preprocess_raw
from tabstar.tabstar_verbalizer import TabSTARVerbalizer
from tabstar.training.trainer import TabStarTrainer


class BaseTabSTAR:
    def __init__(self, model_config=None, trainer_config=None, preprocessor=None):
        # self.model_config = model_config or {}
        # self.trainer_config = trainer_config or {}
        # self.preprocessor = preprocessor
        raise NotImplementedError

    def _prepare(self, X, y):
        x, y = preprocess_raw(x=X, y=y, is_cls=self.is_cls)
        #     self.preprocessor_ = self.preprocessor
        # else:
        #     self.preprocessor_ = self._build_preprocessor()
        #     self.preprocessor_.fit(X, y)
        # return self.preprocessor_.transform(X), y
        raise NotImplementedError

    def _build_preprocessor(self):
        return TabSTARVerbalizer(is_cls=self.is_cls)

    def _build_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        # X_proc, y_proc = self._prepare(X, y)
        # self.model_ = self._build_model()
        # self.trainer_ = TabStarTrainer(self.model_, **self.trainer_config)
        # self.trainer_.train(X_proc, y_proc)
        # return self
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @property
    def is_cls(self) -> bool:
        raise NotImplementedError("Must be implemented in subclass")


class TabSTARClassifier(BaseTabSTAR, BaseEstimator, ClassifierMixin):
    def _build_model(self):
        # return TabStarModel(task="classification", **self.model_config)
        raise NotImplementedError

    def predict_proba(self, X):
        # X_proc = self.preprocessor_.transform(X)
        # return self.model_.predict_proba(X_proc)
        raise NotImplementedError

    @property
    def is_cls(self) -> bool:
        return True


class TabSTARRegressor(BaseTabSTAR, BaseEstimator, RegressorMixin):
    def _build_model(self):
        # return TabStarModel(task="regression", **self.model_config)
        raise NotImplementedError

    def is_cls(self) -> bool:
        return False


