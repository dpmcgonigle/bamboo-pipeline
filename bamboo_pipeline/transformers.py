"""
transformers.py

This module contains the transformers for the PandasFeatureUnion pipelining
"""
from typing import Callable, List

import sklearn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# noinspection PyUnusedLocal
class PandasTransform(TransformerMixin, BaseEstimator):
    """
    Performs a given function on a Pandas DataFrame. Can be a lambda function or Sklearn transformer.
    Sklearn-style transformers are duck-typed and need to implement fit and transform methods.

    Examples:
        Pipeline([
            ('add_five', PandasTransform(lambda X: X+5)),
            ...
        ])
        -   or  -
        Pipeline([
            ('imputer', PandasTransform(
                SimpleImputer(strategy='mean')
            )),
            ...
        ])
    """

    def __init__(self, fn: Callable, axis: int = None) -> None:
        """
        Constructor for PandasTransform.
        fn is a function that can be either a lambda function or a Sklearn transformer.
        axis determines in which direction we traverse the dataframe.  1 will provide each row as a series in apply().
        """
        self.fn: Callable = fn
        self.axis: int = axis

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PandasTransform":
        """
        Fit transformer
        NOTE: Implemented to fit in with TransformerMixin, but currently inactive

        Args:
            X: data
            y: labels

        Returns:
            fitted Transformer object
        """
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform DataFrame

        Args:
            X: data
            y: labels

        Returns:
            transformed_dataframe
        """
        columns = X.columns.values
        #   If the function is a sklearn transformer, call the fit_transform function
        if isinstance(self.fn, sklearn.base.TransformerMixin):
            return pd.DataFrame(
                self.fn.fit_transform(X), columns=columns, index=X.index
            )
        else:
            if self.axis is not None:
                return pd.DataFrame(
                    X.apply(self.fn, axis=self.axis), columns=columns, index=X.index
                )
            else:
                return pd.DataFrame(X.apply(self.fn), columns=columns, index=X.index)


# noinspection PyUnusedLocal
class PandasSubsetSelector(BaseEstimator, TransformerMixin):
    """
    Select a subset of a Pandas Dataframe.
    Sklearn-style transformers are duck-typed and need to implement fit and transform methods.

    Example:
        Pipeline([
            ('selector', PandasSubsetSelector(num_attribs)),
            ...
        ])
    """

    def __init__(self, attribute_names: List[str], dtype: str = None) -> None:
        """
        Create SubsetSelector with given attribute names

        Args:
            attribute_names: list of string column names
            dtype: specify the datatype of the dataframe
        """
        self.attribute_names: List[str] = attribute_names
        self.dtype: str = dtype

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PandasSubsetSelector":
        """
        Fit transformer
        NOTE: Implemented to fit in with TransformerMixin, but currently inactive

        Args:
            X: data
            y: labels

        Returns:
            fitted Transformer object
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by selecting specific columns

        Args:
            X: dataframe

        Returns:
            transformed_dataframe
        """
        if self.dtype:
            return X[self.attribute_names].astype(self.dtype)
        else:
            return X[self.attribute_names]


# noinspection PyUnusedLocal
class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Wrapper class for OneHotEncoder to return Pandas DataFrame instead of scipy sparse matrix.
    Sklearn-style transformers are duck-typed and need to implement fit and transform methods.

    An N x 1 DataFrame column named 'NYHA' that contains values 1.0, 2.0, 3.0 and 4.0 yields:
        An N x 4 DataFrame with columns NYHA_1.0, NYHA_2.0, NYHA_3.0, and NYHA_4.0.

    NOTE: DOES NOT WORK WITH Series OBJECTS.  For instance, passing df['NYHA'] will fail, df[['NYHA']] works

    Example:
        Pipeline([
            ('one_hot_encoder', PandasOneHotEncoder()),
            ...
        ])
    """

    def __init__(self) -> None:
        """Constructor for PandasTransform."""
        self.one_hot_encoder = OneHotEncoder(categories="auto")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PandasOneHotEncoder":
        """
        Fit transformer for one-hot encoding

        Args:
            X: data
            y: labels

        Returns:
            fitted Transformer object
        """
        self.one_hot_encoder.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by one-hot encoding categorical vars

        Args:
            X: dataframe

        Returns:
            transformed_dataframe
        """
        columns: List[str] = []
        for column in X:
            series = X[column]
            #   factorize(): Encode the object as an enumerated type or categorical variable.
            cat_encoded, series_categories = series.factorize()
            series_categories = [
                series.name + "_" + str(x) for x in sorted(series_categories)
            ]
            columns += series_categories
        return pd.DataFrame(
            self.one_hot_encoder.transform(X).toarray(), columns=columns, index=X.index
        )


# noinspection PyUnusedLocal
class PandasLabelEncoder:
    """
    Wrapper class for LabelEncoder to return Pandas DataFrame instead of scipy sparse matrix.
    LabelEncoder can only take a 2-D matrix as input, so we need to wrap it in order to accept a dataframe.
    Sklearn-style transformers are duck-typed and need to implement fit and transform methods.

    Credit to https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
    """

    def __init__(self, columns: List[str] = None) -> None:
        """
        Create LabelEncoder with specified columns
        Args:
            columns: list of column names
        """
        self.columns: List[str] = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PandasLabelEncoder":
        """
        Fit transformer for label encoding
        NOTE: Implemented to fit in with TransformerMixin, but currently inactive

        Args:
            X: data
            y: labels

        Returns:
            fitted Transformer object
        """
        return self  # not relevant here

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms DataFrame X columns with sklearn.preprocessing.LabelEncoder.
        If no columns are specified, transforms all columns in X.

        Args:
            X: dataframe

        Returns:
            transformed_dataframe
        """
        output: pd.DataFrame = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Perform fit and transform on dataset

        Args:
            X: data
            y: labels

        Returns:
            transformed_dataframe from fitted Transformer object
        """
        return self.fit(X, y).transform(X)
