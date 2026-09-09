import numpy as np
import pandas as pd

from fxensemble import (
    EnsembleForecaster,
    ExpandingWindowConfig,
    ForecasterConfig,
    PerformanceEnsembleConfig,
)


def _volatility_panel():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=30, freq="MS")
    assets = ["EUR", "GBP", "JPY", "CHF"]
    rows = []
    for period, date in enumerate(dates):
        for asset_number, asset in enumerate(assets):
            macro = np.sin(period / 5) + asset_number * 0.1
            lagged_vol = 0.08 + 0.015 * asset_number + 0.01 * np.cos(period / 3)
            volatility = np.exp(-2.5 + 0.18 * macro + 0.6 * lagged_vol + rng.normal(0, 0.02))
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "macro": macro,
                    "lagged_vol": lagged_vol,
                    "volatility": volatility,
                    "label_date": date + pd.offsets.MonthBegin(1),
                    "feature_date": date,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[frame.index[7], "macro"] = np.nan
    return frame


def test_end_to_end_volatility_forecast_is_positive_audited_and_deterministic():
    frame = _volatility_panel()
    config = ForecasterConfig(
        models=("ridge", "extra_trees"),
        seed=7,
        n_jobs=2,
        performance_ensemble=PerformanceEnsembleConfig(minimum_history_periods=4),
        model_parameters={
            "ridge": {"alpha": 1.0},
            "extra_trees": {"n_estimators": 20, "max_depth": 4},
        },
    )
    forecaster = EnsembleForecaster(
        config,
        ExpandingWindowConfig(min_train_periods=12),
    )
    kwargs = dict(
        numeric_features=["macro", "lagged_vol"],
        categorical_features=["asset"],
        target_column="volatility",
        date_column="date",
        label_available_column="label_date",
        id_columns=["asset"],
        feature_availability={"macro": "feature_date", "lagged_vol": "feature_date"},
        target_transform=np.log,
        inverse_target_transform=np.exp,
    )

    first = forecaster.fit_predict(frame, **kwargs)
    second = forecaster.fit_predict(frame, **kwargs)

    prediction_columns = [column for column in first.predictions if column.startswith("prediction_")]
    assert (first.predictions[prediction_columns] > 0).all().all()
    assert first.predictions.equals(second.predictions)
    assert first.fitted_models == {}
    assert (first.fold_audit["train_end"] < first.fold_audit["forecast_date"]).all()
    assert (
        first.fold_audit["latest_training_label_available"]
        <= first.fold_audit["forecast_date"]
    ).all()
    assert np.allclose(first.ensemble_weights.groupby("date")["weight"].sum(), 1.0)


def test_future_feature_availability_is_rejected():
    frame = _volatility_panel()
    forecast_date = frame["date"].drop_duplicates().iloc[12]
    frame.loc[frame["date"] == forecast_date, "feature_date"] = forecast_date + pd.Timedelta(days=1)
    forecaster = EnsembleForecaster(
        ForecasterConfig(models=("ols", "ridge")),
        ExpandingWindowConfig(min_train_periods=12, start=str(forecast_date.date())),
    )

    try:
        forecaster.fit_predict(
            frame,
            numeric_features=["macro"],
            target_column="volatility",
            label_available_column="label_date",
            feature_availability={"macro": "feature_date"},
        )
    except ValueError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("Future feature values were accepted")
