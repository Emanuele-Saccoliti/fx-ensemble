"""Public configuration objects for the forecasting framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Any
from collections.abc import Sequence
from numbers import Integral

from sklearn.model_selection import ParameterGrid


@dataclass(frozen=True)
class ExpandingWindowConfig:
    min_train_periods: int
    step_periods: int = 1
    start: str | None = None
    end_exclusive: str | None = None

    def __post_init__(self) -> None:
        if self.min_train_periods < 1 or self.step_periods < 1:
            raise ValueError("Training and step periods must be positive")


@dataclass(frozen=True)
class PreprocessingConfig:
    numeric_imputer: Literal["median", "mean"] = "median"
    add_missing_indicators: bool = True
    scale_linear_models: bool = True


@dataclass(frozen=True)
class PerformanceEnsembleConfig:
    minimum_history_periods: int = 12
    epsilon: float = 1e-8
    loss: Literal["mse", "mae"] = "mse"

    def __post_init__(self) -> None:
        if self.minimum_history_periods < 1 or self.epsilon <= 0:
            raise ValueError("Invalid performance-ensemble configuration")


@dataclass(frozen=True)
class GridSearchConfig:
    """Per-model grids evaluated on the latest inner expanding-window folds."""

    parameter_grids: Mapping[str, Mapping[str, Sequence[Any]]]
    min_train_periods: int
    n_splits: int = 3
    step_periods: int = 1
    scoring: Literal["mse", "mae"] = "mae"
    on_insufficient_history: Literal["raise", "use_fixed"] = "raise"

    def __post_init__(self) -> None:
        for name in ("min_train_periods", "n_splits", "step_periods"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.scoring not in ("mse", "mae"):
            raise ValueError("Grid search scoring must be 'mse' or 'mae'")
        if self.on_insufficient_history not in ("raise", "use_fixed"):
            raise ValueError("on_insufficient_history must be 'raise' or 'use_fixed'")
        if not self.parameter_grids:
            raise ValueError("At least one model parameter grid is required")
        for model, grid in self.parameter_grids.items():
            if not grid:
                raise ValueError(f"Empty parameter grid for {model!r}")
            if any(not isinstance(key, str) or not key or "__" in key for key in grid):
                raise ValueError("Grid keys must be estimator parameter names, without pipeline prefixes")
            ParameterGrid(dict(grid))  # Validate finite, non-empty candidate sequences.


@dataclass(frozen=True)
class ForecasterConfig:
    models: tuple[str, ...] = ("ridge", "elastic_net", "random_forest", "extra_trees")
    seed: int = 0
    n_jobs: int = 1
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    performance_ensemble: PerformanceEnsembleConfig = field(
        default_factory=PerformanceEnsembleConfig
    )
    model_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    grid_search: GridSearchConfig | None = None

    def __post_init__(self) -> None:
        if len(self.models) < 2 or len(set(self.models)) != len(self.models):
            raise ValueError("An ensemble needs at least two unique models")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.grid_search is not None:
            unknown = set(self.grid_search.parameter_grids).difference(self.models)
            if unknown:
                raise ValueError(f"Grid search models are not configured: {sorted(unknown)}")
