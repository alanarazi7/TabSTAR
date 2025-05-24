from sklearn.base import BaseEstimator, ClassifierMixin

from tabstar.arch.arch import TabStarModel
from tabstar.training.trainer import TabStarTrainer


class TabSTARClassifier(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        trainer = TabStarTrainer()
        trainer.train(x=X, y=y)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


