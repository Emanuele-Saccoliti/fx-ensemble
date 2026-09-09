import numpy as np
import pandas as pd

from fxensemble import performance_metrics, regression_metrics


def test_regression_metrics_identify_perfect_forecast():
    frame = pd.DataFrame({"actual": [1.0, -1.0, 2.0], "perfect": [1.0, -1.0, 2.0]})
    metrics = regression_metrics(
        frame, actual_column="actual", prediction_columns=["perfect"]
    ).iloc[0]
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["sign_accuracy"] == 1.0


def test_drawdown_includes_initial_capital():
    metrics = performance_metrics([-0.10, 0.05], periods_per_year=12)
    assert np.isclose(metrics["maximum_drawdown"], -0.10)
