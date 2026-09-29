import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


# ============================================================
# Quantile Clipper
# ============================================================

class QuantileClipper(
    BaseEstimator,
    TransformerMixin,
):
    """
    Learn quantile clipping thresholds from training data.

    Example:
        lower_quantile = 0.01
        upper_quantile = 0.99

    fit(X_train):
        learns the 1st and 99th percentile

    transform(X_val/X_test):
        uses the SAME thresholds learned from X_train
    """

    def __init__(
        self,
        lower_quantile=0.01,
        upper_quantile=0.99,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):

        X = self._to_dataframe(X)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        self.lower_bounds_ = X.quantile(
            self.lower_quantile
        )

        self.upper_bounds_ = X.quantile(
            self.upper_quantile
        )

        return self

    def transform(self, X):

        X = self._to_dataframe(
            X,
            columns=self.feature_names_in_,
        )

        X = X.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1,
        )

        return X

    def get_feature_names_out(
        self,
        input_features=None,
    ):

        if input_features is not None:
            return np.asarray(
                input_features,
                dtype=object,
            )

        return self.feature_names_in_

    @staticmethod
    def _to_dataframe(
        X,
        columns=None,
    ):

        if isinstance(X, pd.DataFrame):
            return X.copy()

        return pd.DataFrame(
            X,
            columns=columns,
        )


# ============================================================
# Log1p Transformer
# ============================================================

class Log1pTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Apply np.log1p().

    Intended for non-negative,
    right-skewed numerical variables.
    """

    def fit(self, X, y=None):

        X_array = np.asarray(X)

        if np.nanmin(X_array) < 0:
            raise ValueError(
                "Log1pTransformer received negative values. "
                "Use log1p only for non-negative features."
            )

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(
                X.columns,
                dtype=object,
            )

        return self

    def transform(self, X):

        return np.log1p(X)

    def get_feature_names_out(
        self,
        input_features=None,
    ):

        if input_features is not None:
            return np.asarray(
                input_features,
                dtype=object,
            )

        return self.feature_names_in_


# ============================================================
# Manual Mapping Transformer
# ============================================================

class CategoryMapper(
    BaseEstimator,
    TransformerMixin,
):
    """
    Map categorical values using dictionaries.

    Example:

    mappings = {
        "risk_level": {
            "low": 0,
            "medium": 1,
            "high": 2
        }
    }
    """

    def __init__(
        self,
        mappings,
    ):
        self.mappings = mappings

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CategoryMapper expects a pandas DataFrame."
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=self.feature_names_in_,
            )

        X = X.copy()

        for col in self.feature_names_in_:

            if col not in self.mappings:
                raise ValueError(
                    f"No mapping found for '{col}'."
                )

            X[col] = X[col].map(
                self.mappings[col]
            )

            # Detect unmapped categories
            if X[col].isna().any():

                raise ValueError(
                    f"Unmapped or missing category "
                    f"found in '{col}'."
                )

        return X.astype(float)

    def get_feature_names_out(
        self,
        input_features=None,
    ):

        if input_features is not None:
            return np.asarray(
                input_features,
                dtype=object,
            )

        return self.feature_names_in_