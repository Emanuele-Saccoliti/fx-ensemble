"""Extensible estimator registry with deterministic built-in models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Mapping

from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge


Factory = Callable[[Mapping[str, Any]], RegressorMixin]


@dataclass(frozen=True)
class RegisteredModel:
    factory: Factory
    linear: bool = False


class ModelRegistry:
    """Registry used by :class:`EnsembleForecaster`; custom estimators are supported."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._models: dict[str, RegisteredModel] = {}
        self._register_defaults()

    def register(self, name: str, factory: Factory, *, linear: bool = False, replace: bool = False) -> None:
        if not name or (name in self._models and not replace):
            raise ValueError(f"Model already registered or invalid: {name!r}")
        self._models[name] = RegisteredModel(factory, linear)

    def create(self, name: str, parameters: Mapping[str, Any] | None = None) -> RegressorMixin:
        if name not in self._models:
            raise KeyError(f"Unknown model {name!r}; available={self.names}")
        return self._models[name].factory(dict(parameters or {}))

    def is_linear(self, name: str) -> bool:
        if name not in self._models:
            raise KeyError(name)
        return self._models[name].linear

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def _register_defaults(self) -> None:
        with_defaults = lambda defaults, supplied: {**defaults, **supplied}
        self.register("ols", lambda p: LinearRegression(**p), linear=True)
        self.register("ridge", lambda p: Ridge(**p), linear=True)
        self.register(
            "elastic_net",
            lambda p: ElasticNet(**with_defaults({"random_state": self.seed}, p)),
            linear=True,
        )
        self.register(
            "random_forest",
            lambda p: RandomForestRegressor(
                **with_defaults({"random_state": self.seed, "n_jobs": 1}, p)
            ),
        )
        self.register(
            "extra_trees",
            lambda p: ExtraTreesRegressor(
                **with_defaults({"random_state": self.seed, "n_jobs": 1}, p)
            ),
        )
        self.register(
            "hist_gradient_boosting",
            lambda p: HistGradientBoostingRegressor(
                **with_defaults({"random_state": self.seed}, p)
            ),
        )
