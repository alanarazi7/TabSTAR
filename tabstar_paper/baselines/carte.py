import os
from typing import Tuple, Any

import torch
from carte_ai import Table2GraphTransformer, CARTEClassifier, CARTERegressor
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.datasets.objects import SupervisedTask

load_dotenv()

class CARTE(TabularModel):

    MODEL_NAME = "CARTE 🗺️"
    SHORT_NAME = "carte"
    USE_VAL_SPLIT = False

    def __init__(self, problem_type: SupervisedTask, device: torch.device, carte_lr_idx: int, verbose: bool = False):
        self.carte_lr_idx = carte_lr_idx
        super().__init__(problem_type=problem_type, device=device, verbose=verbose)
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN not set in .env")
        model_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin", token=token)
        self.carte_preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.carte_preprocessor.fit_transform(x, y=y)

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[Any, Series]:
        x = self.carte_preprocessor.transform(x, y=y)
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
        self.model_.fit(x_train, y_train)
        self.best_val_loss = self.model_.valid_loss_
