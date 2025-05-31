from pandas import Series, DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

from tabstar.preprocessing.nulls import raise_if_null_target


class TabSTARVerbalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # self.num_cols = []
        # self.cat_cols = []
        # self.imputer = None
        # self.scaler = None
        raise NotImplementedError("Initialization is not implemented yet.")

    def fit(self, X: DataFrame, y: Series):
        raise_if_null_target(y)
        # self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        # self.cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
        #
        # self.imputer = SimpleImputer(strategy="mean")
        # self.imputer.fit(X[self.num_cols])
        #
        # self.scaler = StandardScaler()
        # self.scaler.fit(self.imputer.transform(X[self.num_cols]))
        #
        # return self
        raise NotImplementedError("fit is not implemented yet.")

    def transform(self, X: DataFrame, y: Series):
        raise_if_null_target(y)
        # X_transformed = X.copy()
        #
        # # Process numerical columns
        # X_transformed[self.num_cols] = self.imputer.transform(X[self.num_cols])
        # X_transformed[self.num_cols] = self.scaler.transform(X_transformed[self.num_cols])
        #
        # # You can add text/categorical handling here if needed
        # return X_transformed
        raise NotImplementedError("transform is not implemented yet.")

    def fit_transform(self, X: pd.DataFrame, y=None):
        # return self.fit(X, y).transform(X)
        raise NotImplementedError("fit_transform is not implemented yet.")
