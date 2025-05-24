from dataclasses import dataclass, asdict
from typing import Type

from catboost import CatBoost
from sklearn.base import BaseEstimator

BaseModel = BaseEstimator | CatBoost

def init_model(config: dataclass, is_reg: bool, classifier_cls: Type[BaseModel],
               regressor_cls: Type[BaseModel]) -> BaseModel:
    params = asdict(config)
    model_cls = classifier_cls if not is_reg else regressor_cls
    model = model_cls(**params)
    return model