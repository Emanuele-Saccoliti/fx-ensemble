"""Fold-local preprocessing and estimator pipelines."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fxensemble.config import PreprocessingConfig
from fxensemble.registry import ModelRegistry


def make_pipeline(
    model_name: str,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    registry: ModelRegistry,
    preprocessing: PreprocessingConfig,
    model_parameters: dict | None = None,
) -> Pipeline:
    """Create a fresh, unfitted sklearn pipeline for one model and one fold."""

    if not numeric_features and not categorical_features:
        raise ValueError("At least one feature is required")
    transformers = []
    if numeric_features:
        steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing.numeric_imputer,
                    add_indicator=preprocessing.add_missing_indicators,
                    keep_empty_features=True,
                ),
            )
        ]
        if preprocessing.scale_linear_models and registry.is_linear(model_name):
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), list(numeric_features)))
    if categorical_features:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, list(categorical_features)))
    return Pipeline(
        [
            ("preprocessing", ColumnTransformer(transformers, verbose_feature_names_out=False)),
            ("model", registry.create(model_name, model_parameters)),
        ]
    )
