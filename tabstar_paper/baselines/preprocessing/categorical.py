from dataclasses import dataclass
from typing import Optional
from pandas import Series
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

@dataclass
class ModeFiller:
    src: Series
    target: Series
    mode: object

def fill_mode(x_train: Series, x_test: Optional[Series] = None) -> ModeFiller:
    """Fill the test set with the mode of the train set."""
    train_mode = x_train.mode(dropna=True)
    mode_value = train_mode.iloc[0] if not train_mode.empty else np.nan
    x_train_filled = x_train.copy().fillna(mode_value)
    if x_test is not None:
        x_test_filled = x_test.copy().fillna(mode_value)
    else:
        x_test_filled = None
    return ModeFiller(src=x_train_filled, target=x_test_filled, mode=mode_value)

class CategoricalEncoder:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.fitted = False

    def fit(self, x: Series):
        self.encoder.fit(x.values.reshape(-1, 1))
        self.fitted = True

    def transform(self, x: Series) -> Series:
        if not self.fitted:
            raise RuntimeError("Encoder not fitted yet.")
        return Series(self.encoder.transform(x.values.reshape(-1, 1)).astype(int).flatten(), index=x.index)