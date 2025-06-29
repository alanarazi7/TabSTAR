from pandas import Series
from sklearn.preprocessing import OrdinalEncoder


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