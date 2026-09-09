import numpy as np
import pandas as pd

from fxensemble import PerformanceEnsembleConfig, combine_oos_predictions


def _combine(frame):
    return combine_oos_predictions(
        frame,
        model_prediction_columns=["prediction_a", "prediction_b"],
        date_column="date",
        actual_column="actual",
        label_available_column="label_date",
        config=PerformanceEnsembleConfig(minimum_history_periods=2),
    )


def test_performance_weights_use_only_errors_available_at_forecast_time():
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    frame = pd.DataFrame(
        {
            "date": dates,
            "label_date": dates + pd.offsets.MonthBegin(1),
            "actual": [0.0, 1.0, 2.0, 3.0, 4.0],
            "prediction_a": [0.0, 1.0, 2.0, 3.0, 4.0],
            "prediction_b": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    baseline = _combine(frame).weights
    changed = frame.copy()
    changed.loc[changed["date"] >= dates[3], "actual"] = 10_000.0
    counterfactual = _combine(changed).weights

    base_at_t = baseline.loc[baseline["date"] == dates[3], "weight"].to_numpy()
    changed_at_t = counterfactual.loc[
        counterfactual["date"] == dates[3], "weight"
    ].to_numpy()
    assert np.allclose(base_at_t, changed_at_t)
    assert base_at_t[0] > base_at_t[1]


def test_performance_ensemble_has_equal_weight_warmup_and_unit_weight_sums():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    frame = pd.DataFrame(
        {
            "date": dates,
            "label_date": dates + pd.offsets.MonthBegin(1),
            "actual": [1.0, 2.0, 3.0, 4.0],
            "prediction_a": [1.0, 2.0, 3.0, 4.0],
            "prediction_b": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = _combine(frame)

    first = result.weights[result.weights["date"] == dates[0]]
    assert np.allclose(first["weight"], [0.5, 0.5])
    assert np.allclose(result.weights.groupby("date")["weight"].sum(), 1.0)
    assert result.forecasts["prediction_ensemble_performance"].notna().all()
