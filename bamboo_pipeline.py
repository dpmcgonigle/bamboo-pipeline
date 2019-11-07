#! /usr/bin/env python
"""
The Bamboo Pipeline (bamboo_pipeline.py)

This library contains preprocessing functions for Pandas DataFrames based around the concept of using
    PandasFeatureUnion, a function designed to stitch together the results of multiple different Pipelines.
    The key is to return pandas DataFrames instead of (scipy.sparse_matrix, np.array, etc) so that the indexes
    and column names are preserved for the output.

Example:
    # Create the numeric data pre-processing pipeline
    def numeric_pipeline(num_attribs):
        num_pipeline = Pipeline([
            ('selector', PandasSubsetSelector(num_attribs)),
            ('imputer', PandasTransform( SimpleImputer('median') )),
            ('std_scaler', PandasTransform( StandardScaler() ) ),
            ('add_five', PandasTransform(lambda X: X+5))
        ])
        return num_pipeline

    # Create the categorical data pre-processing pipeline
    def categorical_pipeline(cat_attribs):
        cat_pipeline = Pipeline([
            ('selector', PandasSubsetSelector(cat_attribs)),
            ('imputer', PandasTransform( SimpleImputer('most_frequent') )),
            ('cat_encoder', PandasOneHotEncoder())
        ])
        return cat_pipeline

    # Pass columns through without scaling, but imputing most frequent value
    def identity_pipeline(identity_attribs):
        identity_pipeline = Pipeline([
            ('selector', PandasSubsetSelector(identity_attribs)),
            ('imputer', PandasTransform( SimpleImputer('most_frequent') )),
            ('add_five', PandasTransform(lambda X: X+5))
        ])
        return identity_pipeline

    # specify columns each pipeline is going to take from the dataframe	
    num_attribs = [ 'ACEI/ARB', 'Age', 'CABG', ... ]
    cat_attribs = [ 'NYHA' ]
    identity_attribs = [ 'response_num', 'response_cat', 'ID', 'MI' ]

    # Create pipeline
    num_pipeline = numeric_pipeline(num_attribs)
    cat_pipeline = categorical_pipeline(cat_attribs)
    identity_pipeline = identity_pipeline(identity_attribs)

    full_pipeline = PandasFeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ('identity_pipeline', identity_pipeline)
    ])
    #################################################

    # Execute
    crt = full_pipeline.fit_transform(df)
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline, _fit_transform_one, _transform_one
from scipy import sparse
import inspect

#   PandasFeatureUnion class
##################################################################################################
class PandasFeatureUnion(FeatureUnion):
    """
    PandasFeatureUnion is meant to combine data pipelines to create data superset like FeatureUnion, 
    but it maintains the column names and indexes of the Pandas DataFrame.
    Combine pipeline data in Pandas DataFrame using a class that inherits from sklearn.pipeline.FeatureUnion
    Code at https://github.com/marrrcin/pandas-feature-union
    https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union
    
    Example:
        # Create the numeric data pre-processing pipeline
        def numeric_pipeline(num_attribs):
            num_pipeline = Pipeline([
                ('selector', PandasSubsetSelector(num_attribs)),
                ('imputer', PandasTransform( SimpleImputer('median') )),
                ('std_scaler', PandasTransform( StandardScaler() ) ),
                ('add_five', PandasTransform(lambda X: X+5))
            ])
            return num_pipeline

        # Create the categorical data pre-processing pipeline
        def categorical_pipeline(cat_attribs):
            cat_pipeline = Pipeline([
                ('selector', PandasSubsetSelector(cat_attribs)),
                ('imputer', PandasTransform( SimpleImputer('most_frequent') )),
                ('cat_encoder', PandasOneHotEncoder())
            ])
            return cat_pipeline

        # Pass columns through without scaling, but imputing most frequent value
        def identity_pipeline(identity_attribs):
            identity_pipeline = Pipeline([
                ('selector', PandasSubsetSelector(identity_attribs)),
                ('imputer', PandasTransform( SimpleImputer('most_frequent') )),
                ('add_five', PandasTransform(lambda X: X+5))
            ])
            return identity_pipeline

        # specify columns each pipeline is going to take from the dataframe	
        num_attribs = [ 'ACEI/ARB', 'Age', 'CABG', ... ]
        cat_attribs = [ 'NYHA' ]
        identity_attribs = [ 'response_num', 'response_cat', 'ID', 'MI' ]

        # Create pipeline
        num_pipeline = numeric_pipeline(num_attribs)
        cat_pipeline = categorical_pipeline(cat_attribs)
        identity_pipeline = identity_pipeline(identity_attribs)

        full_pipeline = PandasFeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
            ('identity_pipeline', identity_pipeline)
        ])
        #################################################

        # Execute
        crt = full_pipeline.fit_transform(df)
    """
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
#   END PandasFeatureUnion class
##################################################################################################
		
#   PandasTransform class
##################################################################################################
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
    def __init__(self, fn, axis=None):
        """
        Constructor for PandasTransform.
        fn is a function that can be either a lambda function or a Sklearn transformer.
        axis determines in which direction we traverse the dataframe.  1 will provide each row as a series in apply().
        """
        self.fn = fn
        self.axis = axis

    def fit(self, X, y=None):
        """fit: X (required) and y (optional) are Pandas DataFrames."""
        return self

    def transform(self, X, y=None, copy=None):
        """transform: X (required) and y (optional) are Pandas DataFrames. copy is not being used right now."""
        columns = X.columns.values
        #   If the function is a sklearn transformer, call the fit_transform function
        if isinstance(self.fn, sklearn.base.TransformerMixin):
            return pd.DataFrame(self.fn.fit_transform(X), columns=columns, index=X.index)
        else:
            if self.axis is not None:
                return pd.DataFrame(X.apply(self.fn, axis=self.axis), columns=columns, index=X.index)
            else:
                return pd.DataFrame(X.apply(self.fn), columns=columns, index=X.index)
#	END PandasTransform
##################################################################################################

#   PandasSubsetSelector
##################################################################################################
class PandasSubsetSelector(BaseEstimator, TransformerMixin):
    """
    Returns a subset of a Pandas Dataframe.
    Sklearn-style transformers are duck-typed and need to implement fit and transform methods.
    
    Example:
        Pipeline([
            ('selector', PandasSubsetSelector(num_attribs)), 
            ... 
        ])
    """
    def __init__(self, attribute_names, dtype = None):
        """
        Constructor for PandasSubsetSelector.
        
        Params:
            attribute_names(list):      list of column names from the DataFrame for subset selection.
            dtype(str):                 data type of the returned DataFrame columns.
        """
        self.attribute_names = attribute_names
        self.dtype = dtype
        
    def fit(self, X, y=None):
        """fit: X (required) and y (optional) are Pandas DataFrames."""
        return self
        
    def transform(self, X):
        """transform: X is a Pandas DataFrame."""
        if self.dtype:
            return (X[self.attribute_names].astype(self.dtype))
        else:
            return (X[self.attribute_names])
#	END PandasSubsetSelector
##################################################################################################

#	PandasOneHotEncoder()
##################################################################################################
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
    def __init__(self):
        """Constructor for PandasTransform."""
        self.one_hot_encoder = OneHotEncoder(categories='auto')
        
    def fit(self, X, y=None):
        """fit: X (required) and y (optional) are Pandas DataFrames."""
        self.one_hot_encoder.fit(X)
        return self
        
    def transform(self, X):
        """transform: X is a Pandas DataFrame."""
        columns = []
        for column in X:
            series = X[column]
            #   factorize(): Encode the object as an enumerated type or categorical variable.
            cat_encoded, series_categories = series.factorize()
            series_categories = [series.name + '_' + str(x) for x in sorted(series_categories)]
            columns += series_categories
        return pd.DataFrame(self.one_hot_encoder.transform(X).toarray(), columns=columns, index=X.index)
#	END PandasOneHotEncoder
##################################################################################################

#   PandasLabelEncoder
##################################################################################################
class PandasLabelEncoder:
    """
    Wrapper class for LabelEncoder to return Pandas DataFrame instead of scipy sparse matrix.
    LabelEncoder can only take a 2-D matrix as input, so we need to wrap it in order to accept a dataframe.
    Sklearn-style transformers are duck-typed and need to implement fit and transform methods.
    
    Credit to https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
    """
    def __init__(self,columns = None):
        """Constructor for PandasTransform."""
        self.columns = columns

    def fit(self,X,y=None):
        """fit: X (required) and y (optional) are Pandas DataFrames."""
        return self # not relevant here

    def transform(self,X):
        """
        Transforms DataFrame X columns with sklearn.preprocessing.LabelEncoder. 
        If no columns are specified, transforms all columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
#   END PandasLabelEncoder
##################################################################################################