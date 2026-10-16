from typing import List, Optional, Set

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Every output column costs one token in the interaction encoder, so this is a sequence-length budget.
MAX_FEATURES = 200
# Slots of that budget spent on PCA components rather than original columns.
MAX_COMPONENTS = 50
# ...and never more than this share of the budget, so a small budget still keeps mostly originals.
MAX_COMPONENT_FRACTION = 0.25
# Neither the F-statistic nor the PCA basis needs more rows than this.
FIT_MAX_ROWS = 20_000
SEED = 0
COMPONENT_TEMPLATE = "Compressed Feature {i}"


def component_budget(n_features: int) -> int:
    """PCA slots for a table this wide: sqrt growth, so 200 columns buys 10 and 5000+ buys the cap."""
    return int(min(MAX_COMPONENTS, round(np.sqrt(n_features / 2))))


class FeatureSelector:
    """Caps the columns entering the verbalizer at `max_features`.

    Expects constant columns to be gone already (see `constant_columns`). Past the cap the
    budget splits three ways:
    non-numerical columns first, then the numerical columns with the highest F-statistic, then
    `component_budget` slots of PCA over every numerical column that missed the cut, so the tail
    contributes its dominant variance directions instead of nothing. Fit on the training split
    only and replayed by `transform`.

    The F-test ranks each column on its own, by ANOVA between class means for classification and
    by squared Pearson correlation for regression. It is univariate, so it sees neither redundancy
    nor interactions, and its regression side only sees monotone relationships. The PCA tail is
    what hedges that blindness. It was chosen over the wrapper and embedded rankers because it
    fits no model: nothing here is coupled to a model TabArena benchmarks us against, and it costs
    orders of magnitude less on the wide tables that trigger it. Only the ordering of the
    statistic is used, never its p-value, so its normality assumptions do not need to hold.
    """

    def __init__(self, max_features: int = MAX_FEATURES):
        self.max_features = max_features
        self.columns: Optional[List[str]] = None
        self.numerical_kept: Set[str] = set()
        self.compressed_columns: List[str] = []
        self.component_names: List[str] = []
        self.pca: Optional[Pipeline] = None

    @property
    def output_columns(self) -> List[str]:
        return list(self.columns or []) + self.component_names

    @property
    def numerical_columns(self) -> Set[str]:
        return self.numerical_kept | set(self.component_names)

    def fit(self, x: DataFrame, y: Series, is_cls: bool, numerical_features: Set[str]):
        candidates = [str(col) for col in x.columns]
        numerical = [col for col in candidates if col in numerical_features]
        textual = [col for col in candidates if col not in numerical_features]
        if len(candidates) <= self.max_features:
            self._keep(x=x, kept=candidates, numerical=numerical)
            return
        n_components = self._component_slots(n_features=len(candidates), n_numerical=len(numerical), n_rows=len(x))
        textual_kept = textual[:self.max_features - n_components]
        ranked = self._rank_numerical(x=x[numerical], y=y, is_cls=is_cls) if numerical else []
        split = self.max_features - n_components - len(textual_kept)
        numerical_kept, tail = ranked[:split], ranked[split:]
        self._fit_pca(x=x[tail], n_components=n_components, taken=set(x.columns))
        if len(textual_kept) < len(textual):
            print(f"⚠️ {len(textual)} non-numerical features exceed the budget of {self.max_features}; "
                  f"keeping the first {len(textual_kept)}.")
        print(f"🎯 Selecting {len(textual_kept)} non-numerical and {len(numerical_kept)} numerical features "
              f"of {len(candidates)}, plus {len(self.component_names)} compressed from the remaining {len(tail)}.")
        self._keep(x=x, kept=textual_kept + numerical_kept, numerical=numerical_kept)

    def transform(self, x: DataFrame) -> DataFrame:
        if self.columns is None:
            raise ValueError("FeatureSelector must be fitted before transform.")
        missing = [col for col in self.columns + self.compressed_columns if col not in x.columns]
        if missing:
            raise ValueError(f"Columns selected during fit are missing at transform time: {missing}")
        out = x[self.columns].copy()
        if self.pca is not None:
            components = self.pca.transform(_as_float(x[self.compressed_columns]))
            for i, name in enumerate(self.component_names):
                out[name] = components[:, i]
        return out

    def _keep(self, x: DataFrame, kept: List[str], numerical: List[str]):
        kept = set(kept)
        self.columns = [col for col in x.columns if col in kept]
        self.numerical_kept = {col for col in numerical if col in kept}

    def _component_slots(self, n_features: int, n_numerical: int, n_rows: int) -> int:
        """`n_features` is the full width, protected non-numerical columns included: they take
        budget away from the originals too, so a wide table earns the same tail either way."""
        if n_numerical == 0:
            return 0
        # PCA needs strictly more columns and rows than it has components.
        return max(0, min(component_budget(n_features),
                          int(self.max_features * MAX_COMPONENT_FRACTION),
                          n_numerical - 1,
                          min(n_rows, FIT_MAX_ROWS) - 1))

    def _fit_pca(self, x: DataFrame, n_components: int, taken: Set[str]):
        if n_components <= 0 or not len(x.columns):
            return
        self.pca = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", StandardScaler()),
                             ("pca", PCA(n_components=n_components, svd_solver="covariance_eigh"))])
        self.pca.fit(_as_float(_subsample(x)))
        self.compressed_columns = list(x.columns)
        self.component_names = _component_names(n=n_components, taken=taken)

    def _rank_numerical(self, x: DataFrame, y: Series, is_cls: bool) -> List[str]:
        """Numerical column names, most important first."""
        x = _subsample(x)
        values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(_as_float(x))
        target = self._encode_target(y=y.loc[x.index], is_cls=is_cls)
        scores, _ = f_classif(values, target) if is_cls else f_regression(values, target)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-scores, kind="stable")
        return [str(x.columns[i]) for i in order]

    @staticmethod
    def _encode_target(y: Series, is_cls: bool) -> np.ndarray:
        if is_cls:
            return LabelEncoder().fit_transform(y.astype(str))
        return y.to_numpy(dtype=float)


def _subsample(x: DataFrame) -> DataFrame:
    if len(x) <= FIT_MAX_ROWS:
        return x
    return x.sample(n=FIT_MAX_ROWS, random_state=SEED)


def _as_float(x: DataFrame) -> np.ndarray:
    """Numeric matrix with anything unparseable or non-finite as NaN, for the imputer to fill."""
    values = x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def _component_names(n: int, taken: Set[str]) -> List[str]:
    names = [COMPONENT_TEMPLATE.format(i=i + 1) for i in range(n)]
    suffix = 2
    while taken.intersection(names):
        names = [f"{COMPONENT_TEMPLATE.format(i=i + 1)} ({suffix})" for i in range(n)]
        suffix += 1
    return names
