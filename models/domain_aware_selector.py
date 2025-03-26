import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

class DomainAwareSelector(BaseEstimator, TransformerMixin):
    def __init__(self, must_keep_features, selector: SelectFromModel):
        self.must_keep_features = must_keep_features
        self.selector = selector
        
    def fit(self, X, y):
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        self.columns_ = X.columns  # Store the original column names
        self.selector.fit(X, y)
        self.mask_ = self.selector.get_support()
        for f in self.must_keep_features:
            if f in X.columns:
                self.mask_[X.columns.get_loc(f)] = True
        return self

    def transform(self, X):
        # Convert X to DataFrame using stored column names if needed
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=self.columns_)
        return X.loc[:, self.mask_]
