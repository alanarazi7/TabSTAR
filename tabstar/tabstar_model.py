from typing import Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from tabstar.arch.arch import TabStarModel
from tabstar.preprocessing.preprocess import preprocess_raw
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.trainer import TabStarTrainer


class BaseTabSTAR:
    def __init__(self, preprocessor: Optional[TabSTARVerbalizer] = None):
        self.preprocessor_ = preprocessor
        self.model_: Optional[TabStarModel] = None

    def _prepare(self, X, y) -> Tuple[TabSTARData, TabSTARData]:
        x, y = preprocess_raw(x=X, y=y, is_cls=self.is_cls)
        x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        if self.preprocessor_ is None:
            self.preprocessor_ = TabSTARVerbalizer(is_cls=self.is_cls)
            self.preprocessor_.fit(x_train, y_train)
        train_data = self.preprocessor_.transform(x_train, y_train)
        val_data = self.preprocessor_.transform(x_val, y_val)
        return train_data, val_data

    def fit(self, X, y):
        train_data, val_data = self._prepare(X, y)
        trainer = TabStarTrainer()
        trainer.train(train_data, val_data)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @property
    def is_cls(self) -> bool:
        raise NotImplementedError("Must be implemented in subclass")


class TabSTARClassifier(BaseTabSTAR, BaseEstimator, ClassifierMixin):

    def predict_proba(self, X):
        # X_proc = self.preprocessor_.transform(X)
        # return self.model_.predict_proba(X_proc)
        raise NotImplementedError

    @property
    def is_cls(self) -> bool:
        return True


class TabSTARRegressor(BaseTabSTAR, BaseEstimator, RegressorMixin):

    def is_cls(self) -> bool:
        return False


