"""Forecast and investment-performance metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def regression_metrics(
    predictions: pd.DataFrame,
    *,
    actual_column: str,
    prediction_columns: Sequence[str],
) -> pd.DataFrame:
    """Return comparable regression metrics for several forecast columns."""

    missing = ({actual_column} | set(prediction_columns)).difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing metric columns: {sorted(missing)}")
    records: list[dict[str, float | int | str]] = []
    for column in prediction_columns:
        sample = predictions[[actual_column, column]].dropna()
        actual = sample[actual_column].to_numpy(dtype=float)
        forecast = sample[column].to_numpy(dtype=float)
        finite = np.isfinite(actual) & np.isfinite(forecast)
        actual, forecast = actual[finite], forecast[finite]
        if len(actual) == 0:
            raise ValueError(f"No finite observations for {column!r}")
        error = forecast - actual
        sse = float(np.sum(np.square(error)))
        centered = float(np.sum(np.square(actual - actual.mean())))
        zero = float(np.sum(np.square(actual)))
        correlation = (
            float(np.corrcoef(actual, forecast)[0, 1])
            if len(actual) > 1 and actual.std() > 0 and forecast.std() > 0
            else np.nan
        )
        records.append(
            {
                "model": column,
                "n": len(actual),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "r2": 1.0 - sse / centered if centered > 0 else np.nan,
                "r2_vs_zero": 1.0 - sse / zero if zero > 0 else np.nan,
                "correlation": correlation,
                "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(forecast))),
            }
        )
    return pd.DataFrame.from_records(records)


def performance_metrics(
    returns: Sequence[float] | pd.Series,
    *,
    periods_per_year: int = 12,
) -> dict[str, float | int]:
    """Compute annualized return, risk, Sharpe ratio, and maximum drawdown."""

    values = np.asarray(returns, dtype=float)
    invalid_count = np.count_nonzero(np.isnan(values) | np.isinf(values))
    if invalid_count > 0:
        raise ValueError("Returns must be finite and contain no missing values")
    if periods_per_year < 1 or len(values) == 0:
        raise ValueError("Returns and periods_per_year must be non-empty and positive")

    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    total_return = float(wealth[-1] - 1.0)
    annualized_return = float((wealth[-1] ** (periods_per_year / len(values))) - 1.0)
    annualized_volatility = float(np.std(values, ddof=1) * np.sqrt(periods_per_year)) if len(values) > 1 else 0.0
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    return {
        "n_periods": len(values),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "maximum_drawdown": float(drawdown.min()),
    }
