from dataclasses import dataclass, asdict
from typing import Tuple

from catboost import CatBoostRegressor, CatBoostClassifier
from pandas import DataFrame, Series

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

    def transform_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        #     num_cols = sorted(self.numerical_transformers)
        #     x = transform_feature_types(x=x, numerical_features=set(num_cols))
        #     if y is not None:
        #         raise_if_null_target(y)
        #         y = transform_preprocess_y(y=y, scaler=self.target_transformer)
        #     x = verbalize_textual_features(x=x)
        #     x = prepend_target_tokens(x=x, y_name=self.y_name, y_values=self.y_values)
        #     text_cols = [col for col in x.columns if col not in num_cols]
        #     x_txt = x[text_cols + num_cols].copy()
        #     # x_num will hold the numerical features transformed to z-scores, and zero otherwise
        #     x_num = np.zeros(shape=x.shape, dtype=np.float32)
        #     for col in num_cols:
        #         x_txt[col] = transform_numerical_bins(s=x[col], scaler=self.semantic_transformers[col])
        #         idx = x_txt.columns.get_loc(col)
        #         s_num = transform_clipped_z_scores(s=x[col], scaler=self.numerical_transformers[col])
        #         x_num[:, idx] = s_num.to_numpy()
        #     x_txt = x_txt.to_numpy()
        #     data = TabSTARData(d_output=self.d_output, x_txt=x_txt, x_num=x_num, y=y)
        #     return data
        #
        #
        # self.date_transformers = fit_date_encoders(x=x)
        # self.vprint(f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}")
        # x = transform_date_features(x=x, date_transformers=self.date_transformers)
        # x, y = replace_column_names(x=x, y=y)
        # numerical_features = detect_numerical_features(x)
        # self.vprint(f"🔢 Detected {len(numerical_features)} numerical features: {sorted(numerical_features)}")
        # text_features = [col for col in x.columns if col not in numerical_features]
        # self.vprint(f"📝 Detected {len(text_features)} textual features: {sorted(text_features)}")
        # x = transform_feature_types(x=x, numerical_features=numerical_features)
        # self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        # if self.is_cls:
        #     self.d_output = len(self.target_transformer.classes_)
        # else:
        #     self.d_output = 1
        # for col in numerical_features:
        #     self.numerical_transformers[col] = fit_standard_scaler(s=x[col])
        #     self.semantic_transformers[col] = fit_numerical_bins(s=x[col])
        # self.y_name = str(y.name)
        # if self.is_cls:
        #     self.y_values = sorted(self.target_transformer.classes_)
        raise NotImplementedError

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        non_cat_features = set(self.numerical_features).union(set(self.text_transformers))
        cat_features = [i for i, c in enumerate(x_train.columns) if c in non_cat_features]
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features)

