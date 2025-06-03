from pandas import DataFrame, Series

from tabstar.inference.inference import from_x_y

x = None        # TODO: load your X dataset here
y = None        # TODO: load your y series here
is_cls = None   # TODO: set True for classification or False for regression

assert isinstance(x, DataFrame), "x should be a pandas DataFrame"
assert isinstance(y, Series), "y should be a pandas Series"
assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

from_x_y(x=x, y=y, is_cls=is_cls)