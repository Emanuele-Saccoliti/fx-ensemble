"""Public configuration objects for the forecasting framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Any


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
class ForecasterConfig:
    models: tuple[str, ...] = ("ridge", "elastic_net", "random_forest", "extra_trees")
    seed: int = 0
    n_jobs: int = 1
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    performance_ensemble: PerformanceEnsembleConfig = field(
        default_factory=PerformanceEnsembleConfig
    )
    model_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.models) < 2 or len(set(self.models)) != len(self.models):
            raise ValueError("An ensemble needs at least two unique models")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
