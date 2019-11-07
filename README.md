# Bamboo Pipeline

## A scikit-learn / pandas mashup for preprocessing mixed-attribute datasets with ease

### Purpose

This library contains preprocessing transformer functions (inheriting sklearn.base.TransformerMixin) for Pandas DataFrames based around the concept of using PandasFeatureUnion to stitch together the results of multiple different Pipelines. The advantage to this module is to return pandas DataFrames from the pipelined transformer functions instead of the (scipy.sparse_matrix, np.array, etc) native output of sklearn transformers so that the indexes and column names are preserved for the output.

In order for the transformer functions to work with the new PandasFeatureUnion class, we have to make sure the sklearn transformer functions return Pandas DataFrames.  For some of these, we had to create classes that inherit those duck-typed transformers.  PandasOneHotEncoder() mimics OneHotEncoder, creating intuitive column headers based on the values for which they transform.  PandasLabelEncoder() can take an entire DataFrame (or subset of it), rather than LabelEncoder's ability to only take one 2-D column of information.  Other transformer functions can be passed to PandasTransform, such as MinMaxScaler, SimpleImputer, StandardScaler, etc.  You can also pass lambda functions into the pipeline.

### Examples

![Canonical Titanic Dataset](./img/titanic.png)

I have a walkthrough notebook in this repository that you can view in order to see how it all works together.  I have worked through two examples in the notebook, the cononical Titanic dataset, and a Lung Cancer dataset.

* **View Walkthrough Here** [Dataset Walkthrough Notebook](./dataset_walkthroughs.ipynb)

## Creating Custom Transformers

### Lambda Functions and Sklearn Transformers

PandasTransform has been designed so that it can take either sklearn.base.TransformerMixin modules or lambda functions, giving it a wide range of functionality.  if a lambda function is passed, it will utilize DataFrame.apply() with the appropriate axis passed as a parameter.  This gives the capability to perform transformations in one column that are contingent upon another column.  

Lambda Function Examples:

* Change Pclass to 1 for every passenger (row) who is under the age of 25:
<br>PandasTransform(lambda X: X.where(X.index!='Pclass',1) if X['Age']<25 else X, axis=1)
* Impute the maximum value of a column, since SimpleImputer does not provide 'maximum' as a strategy:
<br>PandasTransform(lambda X: X.where(pd.notnull(X),X.max()))
* Map specific values to numbers:
<br>mapping = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5}
<br>PandasTransform( lambda X: mapping[X[0].upper()], axis=1)

Sklearn Function Examples:

* PandasTransform( SimpleImputer(strategy='most_frequent') )
* PandasTransform( sklearn.preprocessing.StandardScaler() )

### Custom Transformers for Transformers that don't play nicely with these modules

There are some transformers that can't be passed to PandasTransform() "off the shelf".  For instance, I have created transformer functions PandasLabelEncoder and PandasOneHotEncoder to wrap the Sklearn functions that do the same.  These functions inherit sklearn.base.BaseEstimator and sklearn.base.TransformerMixin, and this is a requirement for sklearn transformer functions.  They are duck-typed, which means that you have to implement the specific base set of functionalities for it to be considered a transformer function (if it walks like a duck and quacks like a duck...).  These requirements are fit and transform functions.  See the two examples in bamboo_pipeline.py if you would like to see how you might go about creating a new custom transformer for bamboo_pipeline.

## Authors

* **Dan McGonigle** [dpmcgonigle](https://github.com/dpmcgonigle)
* Credit to https://github.com/marrrcin/pandas-feature-union
    https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union
