from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

class TabSTARVerbalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.imputer = None
        self.scaler = None

    def fit(self, X: pd.DataFrame, y=None):
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        self.imputer = SimpleImputer(strategy="mean")
        self.imputer.fit(X[self.num_cols])

        self.scaler = StandardScaler()
        self.scaler.fit(self.imputer.transform(X[self.num_cols]))

        return self

    def transform(self, X: pd.DataFrame):
        X_transformed = X.copy()

        # Process numerical columns
        X_transformed[self.num_cols] = self.imputer.transform(X[self.num_cols])
        X_transformed[self.num_cols] = self.scaler.transform(X_transformed[self.num_cols])

        # You can add text/categorical handling here if needed
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)
