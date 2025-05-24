import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
from carte_ai import Table2GraphTransformer, CARTEClassifier, CARTERegressor
from huggingface_hub import hf_hub_download
import torch
from pandas import DataFrame

from tabular.datasets.tabular_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID
from tabular.evaluation.sklearn_model import init_model
from tabular.models.abstract_sklearn import TabularSklearnModel
from tabular.preprocessing.objects import PreprocessingMethod, SupervisedTask
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.utils import SEED, cprint

#  [2.5, 5, 7.5] × [1e−4, 1e−3]
CARTE_LRS = [0.00025, 0.0025, 0.0005, 0.005, 0.00075, 0.0075]

@dataclass
class CarteHyperparameters:
    device: str
    learning_rate: float
    loss: str
    num_model: int = 5
    n_jobs: int = 1
    random_state: int = SEED
    disable_pbar: bool = False


class CARTE(TabularSklearnModel):

    MODEL_NAME = f"CARTE 🗺️"
    SHORT_NAME = "carte"
    PROCESSING = PreprocessingMethod.CARTE

    def __init__(self, run_name: str, dataset_ids: List[OpenMLDatasetID], device: torch.device,
                 run_num: int, train_examples: int = 0, args: Optional[PretrainArgs] = None,
                 carte_lr_index: Optional[int] = None):
        super().__init__(run_name=run_name, dataset_ids=dataset_ids, device=device, run_num=run_num,
                         train_examples=train_examples, args=args, carte_lr_index=carte_lr_index)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token is None:
            raise ValueError("HUGGINGFACE_HUB_TOKEN not set in .env")
        model_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin", token=token)
        self.preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)


    def initialize_model(self):
        self.model = init_model(config=self.config, is_reg=self.dataset.is_regression,
                                classifier_cls=CARTEClassifier, regressor_cls=CARTERegressor)

    def train(self) -> float:
        # https://github.com/soda-inria/carte
        x_train, y_train = self.load_train()
        cprint(f"Training {self.MODEL_NAME} for {self.dataset.sid}, lr {self.carte_lr_index}, {len(x_train)} examples")
        x_train = self.preprocessor.fit_transform(x_train, y=y_train)
        self.model.fit(x_train, y_train)
        return self.model.valid_loss_

    def preprocess_test(self, x: Any, y: np.ndarray) -> Tuple[Any, np.ndarray]:
        x = self.preprocessor.transform(x)
        return x, y

    def set_config(self) -> CarteHyperparameters:
        if self.carte_lr_index is None:
            cprint(f"Invalid null `carte_lr_index`: {self.carte_lr_index}. Should be 0-{len(CARTE_LRS) - 1}. Set to 0")
            self.carte_lr_index = 0
        lr = CARTE_LRS[self.carte_lr_index]
        task2loss = {SupervisedTask.REGRESSION: 'squared_error',
                     SupervisedTask.BINARY: 'binary_crossentropy',
                     SupervisedTask.MULTICLASS: 'categorical_crossentropy'}
        loss = task2loss[self.dataset.task_type]
        return CarteHyperparameters(device=str(self.device), learning_rate=lr, loss=loss)

    def predict_from_model(self, x: DataFrame, model: Any) -> np.ndarray:
        if self.dataset.is_regression:
            return model.predict(x)
        probs = model.predict_proba(x)
        return probs


# The power transform struggles with some of the variables
BAD_CARTE_DATASETS = {
    # scipy.optimize._optimize.BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points.
    OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
    OpenMLDatasetID.BIN_SOCIAL_JIGSAW_TOXICITY,
    UrlDatasetID.REG_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES,
    KaggleDatasetID.REG_FOOD_WINE_POLISH_MARKET_PRICES,
    KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_PAKISTAN,
    KaggleDatasetID.REG_FOOD_CHOCOLATE_BAR_RATINGS,
    UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER,
    KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY,
    KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23,
    UrlDatasetID.REG_CONSUMER_BIKE_PRICE_BIKEWALE,
    KaggleDatasetID.REG_SOCIAL_KOREAN_DRAMA,
    KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_SAUDI_ARABIA,
    KaggleDatasetID.MUL_FOOD_YELP_REVIEWS,
    OpenMLDatasetID.MUL_HOUSES_MELBOURNE_AIRBNB,
    # ValueError: Length mismatch: Expected axis has 3 elements, new values have 4 elements
    KaggleDatasetID.REG_FOOD_WINE_VIVINO_SPAIN,
}
