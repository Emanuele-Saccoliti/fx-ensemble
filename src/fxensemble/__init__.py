"""Scalable, point-in-time machine-learning ensembles for panel forecasting."""

from fxensemble.config import (
    ExpandingWindowConfig,
    ForecasterConfig,
    PerformanceEnsembleConfig,
    PreprocessingConfig,
)
from fxensemble.ensemble import EnsembleResult, combine_oos_predictions
from fxensemble.forecaster import EnsembleForecaster, ForecastResult
from fxensemble.metrics import performance_metrics, regression_metrics
from fxensemble.portfolio import (
    BacktestResult,
    continuous_backtest,
    ledoit_wolf_covariance,
    mean_variance_weights,
    rank_long_short_weights,
    volatility_scaled_scores,
)
from fxensemble.registry import ModelRegistry
from fxensemble.validation import ExpandingFold, ExpandingWindowSplitter

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "EnsembleForecaster",
    "EnsembleResult",
    "ExpandingFold",
    "ExpandingWindowConfig",
    "ForecastResult",
    "ForecasterConfig",
    "ModelRegistry",
    "PerformanceEnsembleConfig",
    "PreprocessingConfig",
    "ExpandingWindowSplitter",
    "combine_oos_predictions",
    "continuous_backtest",
    "ledoit_wolf_covariance",
    "mean_variance_weights",
    "performance_metrics",
    "rank_long_short_weights",
    "regression_metrics",
    "volatility_scaled_scores",
]
