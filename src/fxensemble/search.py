"""Grid evaluation on precomputed point-in-time inner folds."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from fxensemble.config import GridSearchConfig
from fxensemble.validation import ExpandingFold


def select_parameters(
    train: pd.DataFrame,
    *,
    folds: Sequence[ExpandingFold],
    model_name: str,
    config: GridSearchConfig,
    base_parameters: Mapping[str, Any],
    pipeline_factory: Callable[[dict[str, Any]], Pipeline],
    features: Sequence[str],
    target_column: str,
    target_transform: Callable | None,
    inverse_target_transform: Callable | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Minimize mean per-date loss on original target units; ties use grid order.

    Each candidate receives a fresh pipeline at every inner origin. The caller
    supplies only outer-training rows, including validation labels known by the
    outer origin, and checks feature availability at each inner origin.
    """
    base = dict(base_parameters)
    if len(folds) < config.n_splits:
        if config.on_insufficient_history == "raise":
            raise ValueError(
                f"Grid search for {model_name!r} needs {config.n_splits} inner folds; "
                f"only {len(folds)} available. Increase outer training history, reduce "
                "inner requirements, or set on_insufficient_history='use_fixed'."
            )
        return base, [{
            "candidate_id": -1, "parameters": base, "score": np.nan,
            "split_losses": [], "n_splits": len(folds), "scoring": config.scoring,
            "selected": True, "status": "insufficient_history", "elapsed_seconds": 0.0,
        }]

    records = []
    for candidate_id, candidate in enumerate(ParameterGrid(dict(config.parameter_grids[model_name]))):
        parameters = {**base, **candidate}
        started = perf_counter()
        losses = []
        for fold in folds:
            inner_train = train.iloc[fold.train_indices]
            inner_test = train.iloc[fold.test_indices]
            try:
                pipeline = pipeline_factory(parameters)
                y = inner_train[target_column]
                y = np.asarray(target_transform(y) if target_transform is not None else y, dtype=float)
                if y.shape != (len(inner_train),) or not np.isfinite(y).all():
                    raise ValueError("Target transform returned invalid training values")
                pipeline.fit(inner_train[list(features)], y)
                prediction = pipeline.predict(inner_test[list(features)])
                if inverse_target_transform is not None:
                    prediction = inverse_target_transform(prediction)
                prediction = np.asarray(prediction, dtype=float)
                actual = inner_test[target_column].to_numpy(dtype=float)
                if prediction.shape != actual.shape or not np.isfinite(prediction).all() or not np.isfinite(actual).all():
                    raise ValueError("Non-finite or incorrectly shaped validation values")
                error = prediction - actual
                loss = float(np.mean(np.square(error) if config.scoring == "mse" else np.abs(error)))
                if not np.isfinite(loss):
                    raise ValueError("Validation loss is not finite")
            except Exception as error:
                raise ValueError(
                    f"Grid search model {model_name!r}, candidate {candidate_id} "
                    f"{parameters}, inner date {fold.forecast_date}: {error}"
                ) from error
            losses.append(loss)
        records.append({
            "candidate_id": candidate_id, "parameters": parameters,
            "score": float(np.mean(losses)), "split_losses": losses,
            "n_splits": len(folds), "scoring": config.scoring,
            "selected": False, "status": "evaluated",
            "elapsed_seconds": perf_counter() - started,
        })
    best = min(records, key=lambda row: row["score"])
    best["selected"] = True
    return dict(best["parameters"]), records
