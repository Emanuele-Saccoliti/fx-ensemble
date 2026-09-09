"""Point-in-time forecast combination rules."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fxensemble.config import PerformanceEnsembleConfig


@dataclass(frozen=True)
class EnsembleResult:
    forecasts: pd.DataFrame
    weights: pd.DataFrame


def combine_oos_predictions(
    predictions: pd.DataFrame,
    *,
    model_prediction_columns: Sequence[str],
    date_column: str,
    actual_column: str,
    label_available_column: str,
    config: PerformanceEnsembleConfig | None = None,
) -> EnsembleResult:
    """Add equal and trailing-performance ensembles to OOS predictions.

    Performance weights at date ``t`` use errors only from earlier predictions
    whose labels were observable by ``t``. This prevents future-error leakage.
    """

    cfg = config or PerformanceEnsembleConfig()
    model_columns = list(model_prediction_columns)
    if len(model_columns) < 2 or len(set(model_columns)) != len(model_columns):
        raise ValueError("At least two unique model prediction columns are required")
    required = set(model_columns) | {date_column, actual_column, label_available_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing ensemble columns: {sorted(missing)}")

    out = predictions.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="raise")
    out[label_available_column] = pd.to_datetime(
        out[label_available_column], errors="coerce"
    )
    matrix = out[model_columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Model predictions must be finite")
    equal = np.repeat(1.0 / len(model_columns), len(model_columns))
    out["prediction_ensemble_equal"] = matrix @ equal
    out["prediction_ensemble_performance"] = np.nan

    weight_records: list[dict[str, object]] = []
    for forecast_date in pd.Index(out[date_column].unique()).sort_values():
        past = out.loc[
            (out[date_column] < forecast_date)
            & (out[label_available_column] <= forecast_date)
            & out[actual_column].notna()
        ]
        history_periods = int(past[date_column].nunique())
        weights = equal.copy()
        used_history = history_periods >= cfg.minimum_history_periods and not past.empty
        losses = np.full(len(model_columns), np.nan)
        if used_history:
            errors = past[model_columns].to_numpy(dtype=float) - past[
                actual_column
            ].to_numpy(dtype=float)[:, None]
            losses = (
                np.mean(np.square(errors), axis=0)
                if cfg.loss == "mse"
                else np.mean(np.abs(errors), axis=0)
            )
            inverse = 1.0 / np.maximum(losses, cfg.epsilon)
            weights = inverse / inverse.sum()

        current = out[date_column] == forecast_date
        out.loc[current, "prediction_ensemble_performance"] = (
            out.loc[current, model_columns].to_numpy(dtype=float) @ weights
        )
        for model, weight, loss in zip(model_columns, weights, losses):
            weight_records.append(
                {
                    date_column: pd.Timestamp(forecast_date),
                    "ensemble": "performance",
                    "model": model,
                    "weight": float(weight),
                    "trailing_loss": float(loss) if np.isfinite(loss) else np.nan,
                    "history_periods": history_periods,
                    "used_performance_history": bool(used_history),
                }
            )

    weights_frame = pd.DataFrame.from_records(weight_records)
    sums = weights_frame.groupby(date_column, sort=False)["weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-12):
        raise RuntimeError("Ensemble weights do not sum to one")
    return EnsembleResult(out, weights_frame)
