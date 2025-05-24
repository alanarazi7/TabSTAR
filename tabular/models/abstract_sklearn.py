from os.path import join
from typing import Dict, List, Tuple, Any

import numpy as np
from pandas import DataFrame, Series

from tabular.datasets.torch_dataset import PandasDataset
from tabular.evaluation.metrics import calculate_metric
from tabular.evaluation.predictions import Predictions
from tabular.models.abstract_model import TabularModel
from tabular.preprocessing.objects import SupervisedTask
from tabular.preprocessing.splits import DataSplit
from tabular.preprocessing.target import transform_target
from tabular.preprocessing.trees.categorical import transform_encoder_categorical
from tabular.utils.utils import verbose_print


class TabularSklearnModel(TabularModel):

    def test(self) -> Dict[DataSplit, Predictions]:
        ret = {}
        for split in [DataSplit.DEV, DataSplit.TEST]:
            split_dir = join(self.data_dir, split)
            dataset = PandasDataset(split_dir=split_dir)
            if len(dataset) == 0:
                # TabPFN doesn't have dev split
                continue
            predictions = self.predictions_for_dataset(x=dataset.x, y=dataset.y, task_type=dataset.properties.task_type)
            ret[split] = predictions
        return ret

    def predict(self, x: DataFrame) -> np.ndarray:
        return self.predict_from_model(x=x, model=self.model)

    def predict_from_model(self, x: DataFrame, model: Any) -> np.ndarray:
        verbose_print(f"🔮 Planning to predict over {len(x)} examples, type {type(x)} and shape {x.shape}")
        if self.dataset.is_regression:
            return model.predict(x)
        probs = model.predict_proba(x)
        if self.dataset.is_binary:
            probs = probs[:, 1]
        return probs

    def load_all(self) -> List[DataFrame | Series]:
        train = PandasDataset(split_dir=join(self.data_dir, DataSplit.TRAIN))
        dev = PandasDataset(split_dir=join(self.data_dir, DataSplit.DEV))
        return [train.x, train.y, dev.x, dev.y]

    def load_train(self) -> Tuple[DataFrame, Series]:
        x_train, y_train, x_dev, y_dev = self.load_all()
        assert len(x_dev) == len(y_dev) == 0
        return x_train, y_train

    def predictions_for_dataset(self, x: DataFrame, y: Series | np.ndarray, task_type: SupervisedTask) -> Predictions:
        verbose_print(f"🔮 Planning to predict over {len(x)} examples")
        x, y = self.preprocess_test(x=x, y=y)
        predictions = self.predict(x)
        verbose_print(f"🔮 Predicted {len(predictions)} examples, of type {type(predictions)}")
        metric = calculate_metric(task_type=task_type, y_true=y, y_pred=predictions)
        return Predictions(score=float(metric), predictions=predictions, labels=y)


    def preprocess_test(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        # TODO: this is a bit XGBoost specific, consider excluding some of these and delegating?
        if self.y_scaler is not None:
            y = transform_target(y, transformer=self.y_scaler)
        if self.x_median is not None:
            for col, median in self.x_median.items():
                x[col] = x[col].fillna(median)
        if self.x_encoder is not None:
            for col, encoder in self.x_encoder.items():
                x[col] = transform_encoder_categorical(s=x[col], encoder=encoder)
        return x, y
