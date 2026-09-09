"""Runnable synthetic volatility forecast using fxensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxensemble import (
    EnsembleForecaster,
    ExpandingWindowConfig,
    ForecasterConfig,
    PerformanceEnsembleConfig,
    regression_metrics,
)


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    rows = []
    for period, date in enumerate(dates):
        for number, currency in enumerate(("EUR", "GBP", "JPY", "CHF")):
            macro = np.sin(period / 4) + number / 10
            lagged_volatility = 0.07 + number / 100 + 0.01 * np.cos(period / 3)
            realized_volatility = np.exp(
                -2.7
                + 0.2 * macro
                + 0.8 * lagged_volatility
                + rng.normal(scale=0.025)
            )
            rows.append(
                {
                    "date": date,
                    "currency": currency,
                    "macro": macro,
                    "lagged_volatility": lagged_volatility,
                    "realized_volatility": realized_volatility,
                    "label_available_date": date + pd.offsets.MonthBegin(1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    panel = make_panel()
    forecaster = EnsembleForecaster(
        ForecasterConfig(
            models=("ridge", "extra_trees", "hist_gradient_boosting"),
            seed=5,
            n_jobs=3,
            performance_ensemble=PerformanceEnsembleConfig(
                minimum_history_periods=6
            ),
            model_parameters={"extra_trees": {"n_estimators": 100}},
        ),
        ExpandingWindowConfig(min_train_periods=18),
    )
    result = forecaster.fit_predict(
        panel,
        numeric_features=["macro", "lagged_volatility"],
        categorical_features=["currency"],
        target_column="realized_volatility",
        id_columns=["currency"],
        target_transform=np.log,
        inverse_target_transform=np.exp,
    )
    columns = [column for column in result.predictions if column.startswith("prediction_")]
    metrics = regression_metrics(
        result.predictions,
        actual_column="realized_volatility",
        prediction_columns=columns,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
