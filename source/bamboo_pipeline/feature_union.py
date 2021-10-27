"""
feature_union.py

Wrapper for sklearn.pipeline's FeatureUnion, build to support pandas DataFrames
"""
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class PandasFeatureUnion(FeatureUnion):
    """
    Combine pipeline data in Pandas DataFrame using a class that inherits from sklearn.pipeline.FeatureUnion
    Code at https://github.com/marrrcin/pandas-feature-union
    https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union

    Example:
        # Create sklearn Pipeline objects
        full_pipeline = PandasFeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
            ('identity_pipeline', identity_pipeline)
        ])

        # Execute
        transformed_df = full_pipeline.fit_transform(df)
    """

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series = None, **fit_params
    ) -> Union[pd.DataFrame, np.array]:
        """ Fit all transformers, transform the data and concatenate results

        Args:
            X: data matrix
            y: data labels
            **fit_params:

        Returns:
            transformed_X: pd.DataFrame
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans, X=X, y=y, weight=weight, **fit_params
            )
            for name, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = merge_dataframes_by_column(Xs)
        return Xs

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.array]:
        """ Fit all transformers using X

        Args:
            X: data matrix

        Returns:
            transformed_dataframe
        """
        Xs: List[pd.DataFrame] = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(transformer=trans, X=X, y=None, weight=weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = merge_dataframes_by_column(Xs)
        return Xs


def merge_dataframes_by_column(Xs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DataFrames on column

    Args:
        Xs: DataFrames to merge

    Returns:
        merged_dataframes
    """
    return pd.concat(Xs, axis="columns", copy=False)
