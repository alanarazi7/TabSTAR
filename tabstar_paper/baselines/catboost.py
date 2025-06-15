from dataclasses import dataclass, asdict

from catboost import CatBoostRegressor, CatBoostClassifier
from pandas import DataFrame, Series

from tabstar.preprocessing.splits import split_to_val
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.preprocessing.text_embeddings import fit_text_encoders


@dataclass
class CatBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    early_stopping_rounds: int = 50
    iterations: int = 2000
    od_pval: float = 0.001


class CatBoost(TabularModel):

    MODEL_NAME = "CatBoost 😸"
    SHORT_NAME = "cat"

    def initialize_model(self):
        model_cls = CatBoostClassifier if self.is_cls else CatBoostRegressor
        params = CatBoostDefaultHyperparams()
        self.model_ = model_cls(**asdict(params))

    def fit_internal_preprocessor(self, x: DataFrame, y: Series):
        self.text_transformers = fit_text_encoders(x=x, numerical_features=self.numerical_features, device=self.device)
        # TODO: detect text features and preprocess.
        # TODO: check whether catboost needs to be scaled or something.
        # text_features = [col for col in x.columns if col not in numerical_features]
        # for col in numerical_features:
        #     self.numerical_transformers[col] = fit_standard_scaler(s=x[col])
        #     self.semantic_transformers[col] = fit_numerical_bins(s=x[col])
        raise NotImplementedError("Still to implement CatBoost preprocessing")

    def fit__(self, x, y):
        assert False, "Implement preprocessing"
        x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        cat_features = None
        # cat_features=self.dataset.cat_col_indices
        assert cat_features, "Implelement this"
        transform_texts_to_embeddings(raw=raw, device=device)
        feat_types = {c: str(tp.value) for tp, ls in raw.feature_types.items() for c in ls}
        # TODO: perhaps cat_col_names should become
        cat_col_names = [c for tp, ls in raw.feature_types.items() for c in ls
                         if tp in {FeatureType.BOOLEAN, FeatureType.CATEGORICAL}]
        cat_col_indices = [i for i, c in enumerate(raw.x.columns) if c in cat_col_names]
        y_train = transform_target(y_train, transformer=self.y_scaler)
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features)

