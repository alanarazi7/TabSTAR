from typing import Tuple

from pandas import DataFrame, Series
from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features
from tabstar_paper.baselines.preprocessing.numerical import fit_numerical_median, transform_numerical_features
from tabstar_paper.baselines.preprocessing.categorical import fit_categorical_encoders, transform_categorical_features
from tabstar_paper.benchmarks.constants import CPU_CORES
from tabstar_paper.datasets.objects import SupervisedTask


class RealMLP(TabularModel):
    MODEL_NAME = "RealMLP 🕸"
    SHORT_NAME = "real"

    def initialize_model(self) -> RealMLP_TD_Classifier | RealMLP_TD_Regressor:
        task2metric = {SupervisedTask.BINARY: 'cross_entropy',
                       SupervisedTask.MULTICLASS: '1-auc_ovr',
                       SupervisedTask.REGRESSION: 'rmse'}
        val_metric = task2metric[self.problem_type]
        model_cls = RealMLP_TD_Classifier if self.is_cls else RealMLP_TD_Regressor
        params = {'device': str(self.device), 'n_threads': CPU_CORES, 'val_metric_name': val_metric, 'use_ls': False}
        model = model_cls(**params)
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.numerical_medians = fit_numerical_median(x=x, numerical_features=self.numerical_features)
        self.categorical_encoders = fit_categorical_encoders(x=x, categorical_features=self.categorical_features)
        self.text_transformers = fit_text_encoders(x=x, text_features=self.text_features, device=self.device)
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        x = transform_numerical_features(x=x, numerical_medians=self.numerical_medians)
        x = transform_categorical_features(x=x, categorical_encoders=self.categorical_encoders)
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)

