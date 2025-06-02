from typing import Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from tabstar.arch.arch import TabStarModel
from tabstar.preprocessing.preprocess import preprocess_raw
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.trainer import TabStarTrainer


class BaseTabSTAR:
    def __init__(self, preprocessor: Optional[TabSTARVerbalizer] = None):
        self.preprocessor = preprocessor

    def _prepare(self, X, y) -> Tuple[TabSTARData, TabSTARData]:
        x, y = preprocess_raw(x=X, y=y, is_cls=self.is_cls)
        x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        if isinstance(self.preprocessor, TabSTARVerbalizer):
            self.preprocessor_ = self.preprocessor
        else:
            self.preprocessor_ = self._build_preprocessor()
            self.preprocessor_.fit(x_train, y_train)
        train_data = self.preprocessor_.transform(x_train, y_train)
        val_data = self.preprocessor_.transform(x_val, y_val)
        return train_data, val_data

    def _build_preprocessor(self):
        return TabSTARVerbalizer(is_cls=self.is_cls)

    def _build_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        train_data, val_data = self._prepare(X, y)
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


