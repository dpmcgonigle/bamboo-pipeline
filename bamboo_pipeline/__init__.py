"""
bamboo_pipeline/__init__.py

Pipeline to wrap sklearn.pipeline's FeatureUnion, built to support pandas DataFrames
"""
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from bamboo_pipeline.feature_union import PandasFeatureUnion


class BambooPipeline:
    """

    """

    def __init__(
        self,
        pipelines: Dict[str, Pipeline] = {},
        full_pipeline: PandasFeatureUnion = None,
    ) -> None:
        """

        """
        self.pipelines: Dict[str, Pipeline] = pipelines
        self.full_pipeline: PandasFeatureUnion = full_pipeline

    def add_pipeline(self, pipeline_name: str, **transformers) -> "BambooPipeline":
        """
        Add a new pipeline to the current BambooPipeline

        Args:
            pipeline_name: Name for the pipeline
            **transformers: key-value pairs of "transformer_name"=transformer
                ex: "selector"=PandasSubsetSelector(df_cols), "imputer"=PandasTransform(SimpleImputer('most_frequent'))

        Returns:
            BambooPipeline object with new pipeline
        """
        if pipeline_name in self.pipelines.keys():
            raise ValueError(
                f"BambooPipeline object already contains '{pipeline_name}'"
            )

        transformer_list: List[Tuple[str, TransformerMixin]] = [
            (transformer_name, transformer)
            for transformer_name, transformer in transformers.items()
        ]
        pipelines: Dict[str, Pipeline] = self.pipelines.copy()
        pipelines[pipeline_name] = Pipeline(transformer_list)
        return BambooPipeline(pipelines)

    def fit(self) -> "BambooPipeline":
        """
        Create PipelineFeatureUnion of pipelines to be combined

        Returns:
            New BambooPipeline object with FeatureUnion of pipelines
        """
        full_pipeline = PandasFeatureUnion(
            transformer_list=[
                (pipeline_name, pipeline)
                for pipeline_name, pipeline in self.pipelines.items()
            ]
        )
        return BambooPipeline(pipelines=self.pipelines, full_pipeline=full_pipeline)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Return result of all pipelines concatenated together

        Args:
            dataframe: dataframe to be transformed

        Returns:
            Pandas DataFrame with pipelined data
        """
        return self.full_pipeline.fit_transform(dataframe)

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create PandasFeatureUnion and return result of all pipelines
        This will not update the current object

        Args:
            dataframe: dataframe to be transformed

        Returns:
            Pandas DataFrame with pipelined data
        """
        bamboo_pipeline: "BambooPipeline" = self.fit()
        return bamboo_pipeline.transform(dataframe)
