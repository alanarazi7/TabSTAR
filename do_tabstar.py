from pandas import DataFrame, Series

from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

# --- USER-PROVIDED INPUTS ---
x_train = None  # TODO: load your feature DataFrame here
y_train = None  # TODO: load your target Series here
is_cls = None   # TODO: True for classification, False for regression
x_test = None   # TODO Optional: load your test feature DataFrame (or leave as None)
y_test = None   # TODO Optional: load your test target Series (or leave as None)
# -----------------------------

# Sanity checks
assert isinstance(x_train, DataFrame), "x should be a pandas DataFrame"
assert isinstance(y_train, Series), "y should be a pandas Series"
assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

if x_test is None and y_test is None:
    x_train, x_test, y_train, y_test = split_to_test(x_train, y_train, is_cls=is_cls)

assert isinstance(x_test, DataFrame), "x_test should be a pandas DataFrame"
assert isinstance(y_test, Series), "y_test should be a pandas Series"

tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
tabstar = tabstar_cls()
tabstar.fit(x_train, y_train)
y_pred = tabstar.predict(x_test)