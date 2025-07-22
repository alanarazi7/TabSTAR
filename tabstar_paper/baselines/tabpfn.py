import time
from dataclasses import dataclass
from typing import Tuple, Any

import numpy as np
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.datasets.all_datasets import KaggleDatasetID, OpenMLDatasetID, TabularDatasetID
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features
from tabstar_paper.baselines.preprocessing.numerical import fit_numerical_median, transform_numerical_features
from tabstar_paper.baselines.preprocessing.categorical import fit_categorical_encoders, transform_categorical_features
from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier, TabPFNRegressor as ClientTabPFNRegressor


MAX_SAMPLES = 10_000
MAX_TEST_SIZE = 10_000


# TODO is this really needed? if so we need to move it to a common place
def get_sid(dataset: TabularDatasetID) -> str:
    if isinstance(dataset, OpenMLDatasetID):
        return f"{dataset.value}_{dataset.name}"
    elif isinstance(dataset, (KaggleDatasetID, TabularDatasetID)):
        return f"{dataset.name}".replace('/', '__')
    raise ValueError(f"Invalid dataset type: {dataset}")


@dataclass
class TabPFNDefaultHyperparams:
    random_state: int = SEED


class TabPFN(TabularModel):

    MODEL_NAME = "TabPFN-v2 🤯"
    SHORT_NAME = "pfn"

    def initialize_model(self) -> ClientTabPFNClassifier | ClientTabPFNRegressor:
        model_cls = ClientTabPFNRegressor if not self.is_cls else ClientTabPFNClassifier
        self.vprint(f"☁️ Using API model for TabPFN over dataset")
        model = model_cls()
        return model

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.numerical_medians = fit_numerical_median(x=x, numerical_features=self.numerical_features)
        self.categorical_encoders = fit_categorical_encoders(x=x, categorical_features=self.categorical_features)
        try:
            self.text_transformers = fit_text_encoders(x=x, text_features=self.text_features, device=self.device)
        except Exception as e:
            print('Error fitting text encoders:', e)
        self.text_transformers = {}
        self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        x = transform_numerical_features(x=x, numerical_medians=self.numerical_medians)
        x = transform_categorical_features(x=x, categorical_encoders=self.categorical_encoders)
        x = transform_text_features(x=x, text_encoders=self.text_transformers)
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        if len(x_train) > MAX_SAMPLES:
            self.vprint(f"👇 Using only {MAX_SAMPLES} samples for training TabPFN")
            x_train = x_train[:MAX_SAMPLES]
            y_train = y_train[:MAX_SAMPLES]
        self.vprint(f"Training {self.MODEL_NAME} over {len(x_train)} examples.")
        self.model_.fit(x_train, y_train)

    def predict(self, x: DataFrame) -> np.ndarray:
        x, _ = self.transform_preprocessor(x=x, y=None)
        return self.predict_from_model(x=x, model=self.model_)

    @staticmethod
    def is_valid_dataset(dataset) -> bool:
        if str(dataset) in TABPFN_BLACKLIST:
            print(f"Skipping {dataset} for TabPFN, as it's too big")
            return False
        return True

    def predict_from_model(self, x: DataFrame, model: Any) -> np.ndarray:
        original_dim = x.shape[0]
        batch_size = MAX_TEST_SIZE
        
        # TODO is the following really needed? if so i need to understand it 
        # There is a limit of up to 500,000 cells for inference
        # if self.dataset.sid == get_sid(OpenMLDatasetID.BIN_SOCIAL_JIGSAW_TOXICITY):
        #     # JIGSAW is 45 features, so 10,000 * 45 = 450,000 training cells.
        #     batch_size = 1000


        all_probs = []
        x_batches = [x.iloc[i:i + batch_size] for i in range(0, len(x), batch_size)]
        print(f"Have {len(x_batches)} batches of size {batch_size} for TabPFN")
        for x_batch in x_batches:
            probs = model.predict(x_batch)
            all_probs.append(probs)
            if len(x_batches) > 1:
                to_sleep = 15
                print(f"😴 Sleeping for {to_sleep} seconds to avoid overloading TabPFN API")
                time.sleep(to_sleep)
                print(f"💤 Waking up to continue TabPFN API calls")
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
