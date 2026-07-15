from setuptools import setup, find_packages

setup(
    name="ml-data-validator",
    version="1.0.0",
    description="Automatic data validation framework for ML pipelines",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
)
