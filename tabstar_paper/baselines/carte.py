import os
from typing import Tuple

import torch
from carte_ai import Table2GraphTransformer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pandas import DataFrame, Series

from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.datasets.objects import SupervisedTask

load_dotenv()

class CARTE(TabularModel):

    MODEL_NAME = "CARTE 🗺️"
    SHORT_NAME = "carte"
    USE_VAL_SPLIT = False

    def __init__(self, problem_type: SupervisedTask, device: torch.device, carte_lr_idx: int, verbose: bool = False):
        super().__init__(problem_type=problem_type, device=device, verbose=verbose)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token is None:
            raise ValueError("HUGGINGFACE_HUB_TOKEN not set in .env")
        if not 0 <= carte_lr_idx <= 5:
            raise ValueError(f"Invalid CARTE lr index: {carte_lr_idx}. Should be between 0 and 5.")
        self.carte_lr_idx = carte_lr_idx
        model_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin", token=token)
        self.carte_preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        pass

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        return x, y

    # def initialize_model(self) -> TabICLClassifier:
    #     if not self.is_cls:
    #         raise ValueError("TabICL is only supported for classification tasks for now.")
    #     model = TabICLClassifier(device=str(self.device), checkpoint_version="tabicl-classifier-v1-0208.ckpt")
    #     return model
    #
    # def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
    #     self.model_.fit(x_train, y_train)





'''
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
from carte_ai import Table2GraphTransformer, CARTEClassifier, CARTERegressor
from huggingface_hub import hf_hub_download
import torch
from pandas import DataFrame

from tabstar.constants import SEED
from tabular.datasets.tabular_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID
from tabular.evaluation.sklearn_model import init_model
from tabular.models.abstract_sklearn import TabularSklearnModel
from tabular.preprocessing.objects import PreprocessingMethod, SupervisedTask
from tabular.trainers.pretrain_args import PretrainArgs

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


class CARTE__(TabularSklearnModel):

    def initialize_model(self):
        self.model = init_model(config=self.config, is_reg=self.dataset.is_regression,
                                classifier_cls=CARTEClassifier, regressor_cls=CARTERegressor)

    def train(self) -> float:
        # https://github.com/soda-inria/carte
        x_train, y_train = self.load_train()
        print(f"Training {self.MODEL_NAME} for {self.dataset.sid}, lr {self.carte_lr_index}, {len(x_train)} examples")
        x_train = self.preprocessor.fit_transform(x_train, y=y_train)
        self.model.fit(x_train, y_train)
        return self.model.valid_loss_

    def preprocess_test(self, x: Any, y: np.ndarray) -> Tuple[Any, np.ndarray]:
        x = self.preprocessor.transform(x)
        return x, y

    def set_config(self) -> CarteHyperparameters:
        if self.carte_lr_index is None:
            print(f"Invalid null `carte_lr_index`: {self.carte_lr_index}. Should be 0-{len(CARTE_LRS) - 1}. Set to 0")
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
'''