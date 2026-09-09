"""Point-in-time expanding-window validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxensemble.config import ExpandingWindowConfig


@dataclass(frozen=True)
class ExpandingFold:
    """Integer row positions for a single out-of-sample forecast date."""

    fold_id: int
    forecast_date: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray


class ExpandingWindowSplitter:
    """Generate expanding folds using only labels known at forecast time.

    ``label_available_column`` is the date on which each target became observable.
    A training row is admitted only when its feature date is in the past and its
    label was already available at the forecast origin.
    """

    def __init__(self, config: ExpandingWindowConfig) -> None:
        self.config = config

    def split(
        self,
        frame: pd.DataFrame,
        *,
        date_column: str,
        target_column: str,
        label_available_column: str,
    ) -> list[ExpandingFold]:
        required = {date_column, target_column, label_available_column}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing validation columns: {sorted(missing)}")

        dates = pd.to_datetime(frame[date_column], errors="raise")
        label_dates = pd.to_datetime(frame[label_available_column], errors="coerce")
        candidates = pd.Index(dates.dropna().unique()).sort_values()
        if self.config.start is not None:
            candidates = candidates[candidates >= pd.Timestamp(self.config.start)]
        if self.config.end_exclusive is not None:
            candidates = candidates[candidates < pd.Timestamp(self.config.end_exclusive)]

        folds: list[ExpandingFold] = []
        eligible_number = 0
        for forecast_date in candidates:
            train_mask = (
                (dates < forecast_date)
                & (label_dates <= forecast_date)
                & frame[target_column].notna()
            )
            test_mask = (dates == forecast_date) & frame[target_column].notna()
            train_periods = dates[train_mask].nunique()
            if train_periods < self.config.min_train_periods or not test_mask.any():
                continue
            if eligible_number % self.config.step_periods == 0:
                folds.append(
                    ExpandingFold(
                        fold_id=len(folds),
                        forecast_date=pd.Timestamp(forecast_date),
                        train_indices=np.flatnonzero(train_mask.to_numpy()),
                        test_indices=np.flatnonzero(test_mask.to_numpy()),
                    )
                )
            eligible_number += 1

        if not folds:
            raise ValueError(
                "No valid expanding-window folds; check dates, label availability, "
                "and min_train_periods"
            )
        return folds
