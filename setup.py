"""
setup.py

Setup script for BambooPipeline package
"""
from setuptools import setup

with open("README", "r") as f:
    long_description = f.read()

setup(
    name="bamboo_pipeline",
    version="1.0",
    description="A module for using scikit-learn FeatureUnion/Pipeline with pandas DataFrames",
    license="MIT",
    long_description=long_description,
    author="Dan McGonigle",
    author_email="dpmcgonigle@gmail.com",
    url="https://github.com/dpmcgonigle/bamboo-pipeline",
    packages=["bamboo_pipeline"],  # same as name
    install_requires=["numpy", "pandas", "scikit-learn", "scipy",],
)
