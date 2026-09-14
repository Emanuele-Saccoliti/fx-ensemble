"""High-level ensemble forecaster."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from fxensemble.config import ExpandingWindowConfig, ForecasterConfig
from fxensemble.ensemble import combine_oos_predictions
from fxensemble.preprocessing import make_pipeline
from fxensemble.registry import ModelRegistry
from fxensemble.validation import ExpandingWindowSplitter


ArrayTransform = Callable[[pd.Series | np.ndarray], Any]


@dataclass(frozen=True)
class ForecastResult:
    """Artifacts produced by an expanding-window forecasting run."""

    predictions: pd.DataFrame
    ensemble_weights: pd.DataFrame
    fold_audit: pd.DataFrame
    fitted_models: Mapping[tuple[int, str], Pipeline]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EnsembleForecaster:
    """Fit multiple estimators in leakage-safe expanding windows."""

    def __init__(
        self,
        config: ForecasterConfig,
        validation: ExpandingWindowConfig,
        *,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config
        self.validation = validation
        self.registry = registry or ModelRegistry(seed=config.seed)
        unknown = set(config.models).difference(self.registry.names)
        if unknown:
            raise KeyError(f"Unknown configured models: {sorted(unknown)}")

    def fit_predict(
        self,
        frame: pd.DataFrame,
        *,
        numeric_features: Sequence[str],
        target_column: str,
        date_column: str = "date",
        label_available_column: str = "label_available_date",
        categorical_features: Sequence[str] = (),
        id_columns: Sequence[str] = (),
        feature_availability: Mapping[str, str] | None = None,
        target_transform: ArrayTransform | None = None,
        inverse_target_transform: ArrayTransform | None = None,
        store_fitted_models: bool = False,
    ) -> ForecastResult:
        """Generate one-step OOS forecasts for every valid fold.

        Transform functions are useful for positive targets such as volatility,
        for example ``np.log`` and ``np.exp``. They must be supplied together.
        """

        if (target_transform is None) != (inverse_target_transform is None):
            raise ValueError("target_transform and inverse_target_transform are a pair")
        numeric = list(numeric_features)
        categorical = list(categorical_features)
        identifiers = list(id_columns)
        features = numeric + categorical
        if not features or len(set(features)) != len(features):
            raise ValueError("Features must be non-empty and unique")
        required = set(features + identifiers) | {
            target_column,
            date_column,
            label_available_column,
        }
        availability = dict(feature_availability or {})
        required.update(availability.values())
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing forecast columns: {sorted(missing)}")
        unknown_features = set(availability).difference(features)
        if unknown_features:
            raise ValueError(
                f"Availability declared for non-feature columns: {sorted(unknown_features)}"
            )

        work = frame.copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="raise")
        work[label_available_column] = pd.to_datetime(
            work[label_available_column], errors="coerce"
        )
        for available_column in availability.values():
            work[available_column] = pd.to_datetime(
                work[available_column], errors="coerce"
            )

        splitter = ExpandingWindowSplitter(self.validation)
        folds = splitter.split(
            work,
            date_column=date_column,
            target_column=target_column,
            label_available_column=label_available_column,
        )
        prediction_parts: list[pd.DataFrame] = []
        audit_records: list[dict[str, object]] = []
        stored: dict[tuple[int, str], Pipeline] = {}

        for fold in folds:
            train = work.iloc[fold.train_indices]
            test = work.iloc[fold.test_indices]
            self._check_feature_availability(
                pd.concat([train, test], axis=0),
                forecast_date=fold.forecast_date,
                feature_availability=availability,
            )
            y_train = train[target_column]
            if target_transform is not None:
                y_train = np.asarray(target_transform(y_train), dtype=float)
            else:
                y_train = y_train.to_numpy(dtype=float)
            if not np.isfinite(y_train).all():
                raise ValueError(
                    f"Target transform produced non-finite values in fold {fold.fold_id}"
                )

            def fit_one(model_name: str) -> tuple[str, Pipeline, np.ndarray]:
                pipeline = make_pipeline(
                    model_name,
                    numeric_features=numeric,
                    categorical_features=categorical,
                    registry=self.registry,
                    preprocessing=self.config.preprocessing,
                    model_parameters=dict(
                        self.config.model_parameters.get(model_name, {})
                    ),
                )
                pipeline.fit(train[features], y_train)
                forecast = np.asarray(pipeline.predict(test[features]), dtype=float)
                if inverse_target_transform is not None:
                    forecast = np.asarray(
                        inverse_target_transform(forecast), dtype=float
                    )
                if forecast.shape != (len(test),) or not np.isfinite(forecast).all():
                    raise ValueError(
                        f"Model {model_name!r} returned invalid predictions in "
                        f"fold {fold.fold_id}"
                    )
                return model_name, pipeline, forecast

            fitted = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(fit_one)(name) for name in self.config.models
            )
            columns = identifiers + [date_column, label_available_column, target_column]
            part = test[columns].copy()
            for model_name, pipeline, forecast in fitted:
                part[f"prediction_{model_name}"] = forecast
                if store_fitted_models:
                    stored[(fold.fold_id, model_name)] = pipeline
            prediction_parts.append(part)
            audit_records.append(
                {
                    "fold_id": fold.fold_id,
                    "forecast_date": fold.forecast_date,
                    "train_rows": len(train),
                    "train_periods": train[date_column].nunique(),
                    "test_rows": len(test),
                    "train_start": train[date_column].min(),
                    "train_end": train[date_column].max(),
                    "latest_training_label_available": train[
                        label_available_column
                    ].max(),
                    "models": ",".join(self.config.models),
                }
            )

        predictions = pd.concat(prediction_parts, ignore_index=True)
        prediction_columns = [f"prediction_{name}" for name in self.config.models]
        combined = combine_oos_predictions(
            predictions,
            model_prediction_columns=prediction_columns,
            date_column=date_column,
            actual_column=target_column,
            label_available_column=label_available_column,
            config=self.config.performance_ensemble,
        )
        sort_columns = [date_column] + identifiers
        forecasts = combined.forecasts.sort_values(sort_columns).reset_index(drop=True)
        return ForecastResult(
            predictions=forecasts,
            ensemble_weights=combined.weights,
            fold_audit=pd.DataFrame.from_records(audit_records),
            fitted_models=stored,
            metadata={
                "forecaster": asdict(self.config),
                "validation": asdict(self.validation),
                "numeric_features": numeric,
                "categorical_features": categorical,
                "id_columns": identifiers,
                "target_column": target_column,
                "date_column": date_column,
                "label_available_column": label_available_column,
                "feature_availability": availability,
                "target_transform": None if target_transform is None else getattr(
                    target_transform, "__qualname__", getattr(target_transform, "__name__", type(target_transform).__name__)
                ),
                "inverse_target_transform": None if inverse_target_transform is None else getattr(
                    inverse_target_transform, "__qualname__", getattr(inverse_target_transform, "__name__", type(inverse_target_transform).__name__)
                ),
            },
        )

    @staticmethod
    def _check_feature_availability(
        rows: pd.DataFrame,
        *,
        forecast_date: pd.Timestamp,
        feature_availability: Mapping[str, str],
    ) -> None:
        for feature, available_column in feature_availability.items():
            populated = rows[feature].notna()
            invalid = populated & (
                rows[available_column].isna()
                | (rows[available_column] > forecast_date)
            )
            if invalid.any():
                raise ValueError(
                    f"Feature {feature!r} contains values unavailable at "
                    f"forecast date {forecast_date.date()}"
                )
