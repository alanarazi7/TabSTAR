from dataclasses import dataclass, asdict

from catboost import CatBoostRegressor, CatBoostClassifier
from pandas import DataFrame, Series

from tabstar.preprocessing.splits import split_to_val
from tabstar_paper.baselines.abstract_model import TabularModel


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
        # numerical_features = detect_numerical_features(x)
        # text_features = [col for col in x.columns if col not in numerical_features]
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
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features)


# class CatBoostOptuna(CatBoost):
#     def train(self):
#         study.optimize(self.objective, n_jobs=OPTUNA_CPU, timeout=OPTUNA_BUDGET,
#                        catch=(CatBoostError,))
#         cprint(f"Done studying, did {len(study.trials)} runs 🤓")
#         self.config = CatBoostTunedHyperparams(**study.best_params)
#         wandb.log({"optuna_best_params": asdict(self.config), "optuna_n_trials": len(study.trials)})
#         cprint(f"✅ Best params: {self.config}")
#         self.initialize_model()
#         assert self.model is not None
#         verbose_print(f"Training {self.MODEL_NAME} FULL model for dataset {self.dataset.sid}")
#         x_train, y_train = self.load_train()
#         if self.task_type == SupervisedTask.REGRESSION:
#             self.y_scaler = fit_standard_scaler(y_train)
#             y_train = transform_target(y_train, transformer=self.y_scaler)
#         self.model.fit(x_train, y_train, logging_level=LOG_LEVEL, use_best_model=True,
#                        cat_features=self.dataset.cat_col_indices)
#
#     def objective(self, trial: Trial) -> float:
#         # Hyperparam search as suggested by TabPFN-v2 paper: https://www.nature.com/articles/s41586-024-08328-6.pdf
#         lr = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
#         random_strength = trial.suggest_int("random_strength", 1, 20)
#         l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
#         bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
#         leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 20)
#         iterations = trial.suggest_int("iterations", 100, 4000)
#         trial_config = CatBoostTunedHyperparams(iterations=iterations,
#                                                 learning_rate=lr,
#                                                 random_strength=random_strength,
#                                                 l2_leaf_reg=l2_leaf_reg,
#                                                 bagging_temperature=bagging_temperature,
#                                                 leaf_estimation_iterations=leaf_estimation_iterations)
#
#         x, y = self.load_train()
#         splitter = get_kfold_splitter(is_regression=self.dataset.is_regression)
#         fold_scores = []
#         for f, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
#             verbose_print(f"Training fold {f}")
#             x_train, y_train, x_dev, y_dev = make_train_dev_splits(x=x, y=y, train_idx=train_idx, val_idx=val_idx)
#             if self.task_type == SupervisedTask.REGRESSION:
#                 y_train, y_dev = standardize_y_train_test(y_train, y_dev)
#             fold_model = init_model(config=trial_config, is_reg=self.dataset.is_regression,
#                                     classifier_cls=CatBoostClassifier, regressor_cls=CatBoostRegressor)
#             fold_model.fit(x_train, y_train, eval_set=(x_dev, y_dev), logging_level=LOG_LEVEL, use_best_model=True,
#                            cat_features=self.dataset.cat_col_indices)
#             predictions = self.predict_from_model(x_dev, model=fold_model)
#             metric = calculate_metric(task_type=self.task_type, y_true=y_dev, y_pred=predictions)
#             fold_scores.append(metric)
#         avg_score = float(np.mean(fold_scores))
#         return avg_score
