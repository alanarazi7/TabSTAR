from pandas import DataFrame, Series
from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

from tabstar.training.devices import CPU_CORES
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.datasets.objects import SupervisedTask


class RealMLP(TabularModel):
    MODEL_NAME = "RealMLP 🕸"
    SHORT_NAME = "real"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> RealMLP_TD_Classifier | RealMLP_TD_Regressor:
        task2metric = {SupervisedTask.BINARY: 'cross_entropy',
                       SupervisedTask.MULTICLASS: '1-auc_ovr',
                       SupervisedTask.REGRESSION: 'rmse'}
        val_metric = task2metric[self.problem_type]
        model_cls = RealMLP_TD_Classifier if self.is_cls else RealMLP_TD_Regressor
        params = {'device': str(self.device), 'n_threads': CPU_CORES, 'val_metric_name': val_metric, 'use_ls': False}
        model = model_cls(**params)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)

