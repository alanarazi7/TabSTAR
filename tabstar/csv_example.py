from pandas import DataFrame, Series

from tabstar.inference.inference import from_x_y

x = None        # TODO: load your X dataset here
y = None        # TODO: load your y series here
is_cls = None   # TODO: set True for classification or False for regression
x_test = None   # TODO Optional: load your X test dataset here. If you don't, we'll split it randomly.
y_test = None   # TODO Optional: load your y test series here. If you don't, we'll split it randomly.

assert isinstance(x, DataFrame), "x should be a pandas DataFrame"
assert isinstance(y, Series), "y should be a pandas Series"
assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

y_pred = from_x_y(x=x, y=y, is_cls=is_cls, x_test=x_test, y_test=y_test)