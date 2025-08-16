from typing import Tuple, Any, Optional

import numpy as np
import torch
from carte_ai import Table2GraphTransformer, CARTEClassifier, CARTERegressor
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.datasets.all_datasets import KaggleDatasetID, OpenMLDatasetID, UrlDatasetID
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.constants import HF_TOKEN
from tabstar_paper.datasets.objects import SupervisedTask

load_dotenv()

class CARTE(TabularModel):

    MODEL_NAME = "CARTE 🗺️"
    SHORT_NAME = "carte"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = False

    def __init__(self, problem_type: SupervisedTask, device: torch.device, carte_lr_idx: int, verbose: bool = False):
        self.carte_lr_idx = carte_lr_idx
        super().__init__(problem_type=problem_type, device=device, verbose=verbose)
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN not set in .env")
        model_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin", token=HF_TOKEN)
        self.carte_preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        y = y.copy()
        if not self.is_cls:
            y = y.to_numpy().reshape(-1, 1)
        self.carte_preprocessor.fit(x, y=y)

    def transform_internal_preprocessor(self, x: DataFrame, y: Optional[Series]) -> Tuple[Any, Series]:
        y_npy = y.copy().to_numpy() if y is not None else None
        x = self.carte_preprocessor.transform(x, y=y_npy)
        return x, y

    def initialize_model(self) -> CARTEClassifier | CARTERegressor:
        model_cls = CARTEClassifier if self.is_cls else CARTERegressor
        task2loss = {SupervisedTask.REGRESSION: 'squared_error',
                     SupervisedTask.BINARY: 'binary_crossentropy',
                     SupervisedTask.MULTICLASS: 'categorical_crossentropy'}
        loss_func = task2loss[self.problem_type]
        carte_lrs = [0.00025, 0.0025, 0.0005, 0.005, 0.00075, 0.0075]
        if not 0 <= self.carte_lr_idx <= 5:
            raise ValueError(f"Invalid CARTE lr index: {self.carte_lr_idx}. Should be between 0 and 5.")
        lr = carte_lrs[self.carte_lr_idx]
        params = {
            "device": str(self.device),
            "learning_rate": lr,
            "loss": loss_func,
            "num_model": 5,
            "n_jobs": 1,
            "random_state": SEED,
            "disable_pbar": False,
        }
        model = model_cls(**params)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        y_train_array = y_train.copy().to_numpy()
        self.model_.fit(X=x_train, y=y_train_array)
        self.best_val_loss = self.model_.valid_loss_

    def predict(self, x: DataFrame) -> np.ndarray:
        x, _ = self.transform_preprocessor(x=x, y=None)
        if not self.is_cls:
            return self.model_.predict(x)
        else:
            return self.model_.predict_proba(x)


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