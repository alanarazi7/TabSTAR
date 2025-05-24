import time

from typing import Any

import numpy as np
from pandas import DataFrame

from tabular.datasets.tabular_datasets import OpenMLDatasetID, TabularDatasetID, KaggleDatasetID, get_sid
from tabular.preprocessing.splits import MAX_TEST_SIZE
from tabular.utils.utils import cprint

from tabular.models.abstract_sklearn import TabularSklearnModel
from tabular.preprocessing.objects import PreprocessingMethod
from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier, TabPFNRegressor as ClientTabPFNRegressor


MAX_SAMPLES = 10_000

class TabPFNv2(TabularSklearnModel):

    MODEL_NAME = "TabPFN-v2 🤯"
    SHORT_NAME = "pfn"
    PROCESSING = PreprocessingMethod.TABPFNV2

    def initialize_model(self):
        model_cls = ClientTabPFNRegressor if self.dataset.is_regression else ClientTabPFNClassifier
        cprint(f"☁️ Using API model for TabPFN over textual dataset {self.dataset}")
        self.model = model_cls()

    def train(self):
        x_train, y_train = self.load_train()
        if len(x_train) > MAX_SAMPLES:
            cprint(f"👇 Using only {MAX_SAMPLES} samples for training TabPFN")
            x_train = x_train[:MAX_SAMPLES]
            y_train = y_train[:MAX_SAMPLES]
        cprint(f"Training {self.MODEL_NAME} over {len(x_train)} examples.")
        self.model.fit(x_train, y_train)

    def set_config(self) -> None:
        return None

    @staticmethod
    def is_valid_dataset(dataset: TabularDatasetID) -> bool:
        if dataset in TABPFN_BLACKLIST:
            print(f"Skipping {dataset} for TabPFN, as it's too big")
            return False
        return True

    def predict_from_model(self, x: DataFrame, model: Any) -> np.ndarray:
        original_dim = x.shape[0]
        batch_size = MAX_TEST_SIZE
        # There is a limit of up to 500,000 cells for inference
        if self.dataset.sid == get_sid(OpenMLDatasetID.BIN_SOCIAL_JIGSAW_TOXICITY):
            # JIGSAW is 45 features, so 10,000 * 45 = 450,000 training cells.
            batch_size = 1000
        all_probs = []
        x_batches = [x.iloc[i:i + batch_size] for i in range(0, len(x), batch_size)]
        cprint(f"Have {len(x_batches)} batches of size {batch_size} for TabPFN")
        for x_batch in x_batches:
            probs = super().predict_from_model(x=x_batch, model=model)
            all_probs.append(probs)
            if len(x_batches) > 1:
                to_sleep = 15
                cprint(f"😴 Sleeping for {to_sleep} seconds to avoid overloading TabPFN API")
                time.sleep(to_sleep)
                cprint(f"💤 Waking up to continue TabPFN API calls")
        all_probs = np.concatenate(all_probs)
        assert len(all_probs) == original_dim, f"Expected {original_dim} predictions, got {len(all_probs)}"
        return all_probs

TABPFN_BLACKLIST = {
    # Has too many classes
    OpenMLDatasetID.MUL_FOOD_WINE_REVIEW,
    # "Your client issued a request that was too large."
    OpenMLDatasetID.MUL_HOUSES_MELBOURNE_AIRBNB,
    # There is a limit of up to 500,000 cells for inference
    KaggleDatasetID.REG_CONSUMER_CAR_PRICE_CARDEKHO,
    KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23,
}
