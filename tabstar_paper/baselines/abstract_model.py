from typing import Tuple

from pandas import DataFrame, Series

from tabstar.preprocessing.splits import split_to_val


class TabularModel:

    MODEL_NAME: str
    SHORT_NAME: str

    def __init__(self, is_cls: bool):
        self.is_cls = is_cls
        self.model_ = self.initialize_model()
        # self.run_name = run_name
        # self.dataset_ids = dataset_ids
        # self.device = device
        # self.run_num = run_num
        # self.train_examples = train_examples
        # self.data_dirs: List[str] = self.initialize_data_dirs()
        # self.datasets: List[DatasetProperties] = [get_properties(d) for d in self.data_dirs]
        # self.model: Optional[Any] = None
        # self.config = self.set_config()
        # # For processing
        # self.y_scaler: Optional[StandardScaler] = None
        # self.x_median: Optional[Dict[str, float]] = None
        # self.x_encoder: Optional[Dict[str, ColumnLabelEncoder]] = None

    def initialize_model(self):
        raise NotImplementedError("Initialize model method not implemented yet")

    def fit(self, x: DataFrame, y: Series):
        # TODO: for methods which don't require internal split_to_val, skip this step
        x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        self.fit_preprocessor(x_train=x_train, y_train=y_train)
        x_train, y_train = self.transform_preprocessor(x=x_train, y=y_train)
        x_val, y_val = self.transform_preprocessor(x=x_val, y=y_val)
        self.fit_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    def fit_preprocessor(self, x_train: DataFrame, y_train: Series):
        raise NotImplementedError("Fit preprocessor method not implemented yet")

    def transform_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        raise NotImplementedError("Transform preprocessor method not implemented yet")

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        raise NotImplementedError("Fit model method not implemented yet")



    # def test(self) -> Dict[DataSplit, Predictions]:
    #     ret = {}
    #     for split in [DataSplit.DEV, DataSplit.TEST]:
    #         split_dir = join(self.data_dir, split)
    #         dataset = PandasDataset(split_dir=split_dir)
    #         if len(dataset) == 0:
    #             # TabPFN doesn't have dev split
    #             continue
    #         predictions = self.predictions_for_dataset(x=dataset.x, y=dataset.y, task_type=dataset.properties.task_type)
    #         ret[split] = predictions
    #     return ret
    #
    # def predict(self, x: DataFrame) -> np.ndarray:
    #     return self.predict_from_model(x=x, model=self.model)
    #
    # def predict_from_model(self, x: DataFrame, model: Any) -> np.ndarray:
    #     verbose_print(f"🔮 Planning to predict over {len(x)} examples, type {type(x)} and shape {x.shape}")
    #     if self.dataset.is_regression:
    #         return model.predict(x)
    #     probs = model.predict_proba(x)
    #     if self.dataset.is_binary:
    #         probs = probs[:, 1]
    #     return probs
    #
    # def load_all(self) -> List[DataFrame | Series]:
    #     train = PandasDataset(split_dir=join(self.data_dir, DataSplit.TRAIN))
    #     dev = PandasDataset(split_dir=join(self.data_dir, DataSplit.DEV))
    #     return [train.x, train.y, dev.x, dev.y]
    #
    # def load_train(self) -> Tuple[DataFrame, Series]:
    #     x_train, y_train, x_dev, y_dev = self.load_all()
    #     assert len(x_dev) == len(y_dev) == 0
    #     return x_train, y_train
    #
    # def predictions_for_dataset(self, x: DataFrame, y: Series | np.ndarray, task_type: SupervisedTask) -> Predictions:
    #     verbose_print(f"🔮 Planning to predict over {len(x)} examples")
    #     x, y = self.preprocess_test(x=x, y=y)
    #     predictions = self.predict(x)
    #     verbose_print(f"🔮 Predicted {len(predictions)} examples, of type {type(predictions)}")
    #     metric = calculate_metric(task_type=task_type, y_true=y, y_pred=predictions)
    #     return Predictions(score=float(metric), predictions=predictions, labels=y)
    #
    #
    # def preprocess_test(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    #     # TODO: this is a bit XGBoost specific, consider excluding some of these and delegating?
    #     if self.y_scaler is not None:
    #         y = transform_target(y, transformer=self.y_scaler)
    #     if self.x_median is not None:
    #         for col, median in self.x_median.items():
    #             x[col] = x[col].fillna(median)
    #     if self.x_encoder is not None:
    #         for col, encoder in self.x_encoder.items():
    #             x[col] = transform_encoder_categorical(s=x[col], encoder=encoder)
    #     return x, y

