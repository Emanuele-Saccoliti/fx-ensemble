from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from fxensemble import (
    EnsembleForecaster, ExperimentStore, ExpandingWindowConfig, ForecasterConfig,
    GridSearchConfig, ModelRegistry,
)
from fxensemble.preprocessing import make_pipeline


def panel():
    dates = pd.date_range("2020-01-01", periods=11, freq="MS")
    frame = pd.DataFrame({
        "date": np.repeat(dates, 2), "asset": ["a", "b"] * len(dates),
        "x": np.arange(22, dtype=float),
        "target": 1 + 2 * np.arange(22, dtype=float),
        "label_available_date": np.repeat(dates + pd.offsets.MonthBegin(1), 2),
    })
    # One old label is unavailable even at the outer origin; another is delayed
    # until the last inner origin. Both must respect their publication dates.
    frame.loc[4, "label_available_date"] = dates[-1] + pd.offsets.MonthBegin(2)
    frame.loc[10, "label_available_date"] = dates[-2]
    return frame


def run(frame, search, *, models=("ols", "ridge"), parameters=None, registry=None, n_jobs=1, **kwargs):
    forecaster = EnsembleForecaster(
        ForecasterConfig(models=models, model_parameters=parameters or {}, grid_search=search, n_jobs=n_jobs),
        ExpandingWindowConfig(min_train_periods=5, start=str(frame.date.max().date())),
        registry=registry,
    )
    return forecaster.fit_predict(
        frame, numeric_features=["x"], target_column="target", id_columns=["asset"],
        store_fitted_models=True, **kwargs,
    )


def test_grid_matches_manual_temporal_validation_and_refits_winner(tmp_path):
    frame = panel()
    search = GridSearchConfig({"ridge": {"alpha": [0.1, 100.]}}, min_train_periods=3)
    result = run(frame, search, parameters={"ridge": {"fit_intercept": False}})
    scores = result.search_results
    assert len(scores) == 2 and scores.selected.sum() == 1
    audit = result.search_audit
    assert len(audit) == 3
    assert (audit.train_end < audit.validation_date).all()
    assert (audit.latest_training_label_available <= audit.validation_date).all()
    assert (audit.latest_validation_label_available <= audit.forecast_date).all()
    assert (audit.validation_date < audit.forecast_date).all()
    assert list(audit.validation_rows) == [2, 2, 2]
    outer_train = frame[(frame.date < frame.date.max()) & (frame.label_available_date <= frame.date.max())]
    for row in scores.itertuples():
        losses = []
        for date in audit.validation_date:
            inner_train = outer_train[(outer_train.date < date) & (outer_train.label_available_date <= date)]
            inner_test = outer_train[outer_train.date == date]
            pipeline = make_pipeline(
                "ridge", numeric_features=["x"], registry=ModelRegistry(),
                preprocessing=ForecasterConfig().preprocessing, model_parameters=row.parameters,
            )
            pipeline.fit(inner_train[["x"]], inner_train.target)
            losses.append(np.abs(pipeline.predict(inner_test[["x"]]) - inner_test.target).mean())
        assert np.allclose(row.split_losses, losses)
        assert np.isclose(row.score, np.mean(losses))
    winner = scores.loc[scores.selected].iloc[0]
    fitted = result.fitted_models[(0, "ridge")]
    assert fitted.named_steps["model"].alpha == winner.parameters["alpha"]
    assert not fitted.named_steps["model"].fit_intercept
    scaler = fitted.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(outer_train)
    assert set(scores.model) == {"ridge"}  # OLS retains its fixed configuration.
    store = ExperimentStore(tmp_path)
    saved = store.save("tuned", result)
    loaded = store.load("tuned")
    pd.testing.assert_frame_equal(loaded.search_results, result.search_results)
    pd.testing.assert_frame_equal(loaded.search_audit, result.search_audit)
    assert "Temporal grid search" in (saved / "report.html").read_text()
    assert loaded.metadata["configuration"]["forecaster"]["grid_search"]["scoring"] == "mae"


def test_outer_targets_and_unreleased_labels_cannot_change_selection():
    frame = panel()
    search = GridSearchConfig({"ridge": {"alpha": [0.1, 100.]}}, min_train_periods=3)
    first = run(frame, search)
    changed = frame.copy()
    changed.loc[(changed.date == changed.date.max()) | (changed.label_available_date > changed.date.max()), "target"] = 999999.
    second = run(changed, search)
    pd.testing.assert_frame_equal(
        first.search_results.drop(columns="elapsed_seconds"),
        second.search_results.drop(columns="elapsed_seconds"),
    )
    assert np.allclose(first.predictions.prediction_ridge, second.predictions.prediction_ridge)


def test_preprocessing_is_fitted_separately_inside_each_split(monkeypatch):
    import fxensemble.forecaster as module
    pipelines = []
    def capture(*args, **kwargs):
        pipeline = make_pipeline(*args, **kwargs)
        if args[0] == "ridge":
            pipelines.append(pipeline)
        return pipeline
    monkeypatch.setattr(module, "make_pipeline", capture)
    frame = panel()
    frame.loc[0, "x"] = np.nan
    result = run(frame, GridSearchConfig({"ridge": {"alpha": [1.]}}, min_train_periods=3))
    assert len(pipelines) == 4  # Three fresh inner pipelines and the outer refit.
    outer = frame[(frame.date < frame.date.max()) & (frame.label_available_date <= frame.date.max())]
    for pipeline, date in zip(pipelines, result.search_audit.validation_date):
        subset = outer[(outer.date < date) & (outer.label_available_date <= date)]
        numeric = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
        assert numeric.named_steps["imputer"].statistics_[0] == subset.x.median()
        assert numeric.named_steps["scaler"].n_samples_seen_ == len(subset)


def test_inner_feature_availability_is_checked():
    frame = panel()
    frame["release"] = frame.date
    frame.loc[frame.date == frame.date.unique()[-2], "release"] = frame.date.max()
    with pytest.raises(ValueError, match="unavailable"):
        run(frame, GridSearchConfig({"ridge": {"alpha": [1.]}}, 3), feature_availability={"x": "release"})


def test_insufficient_history_errors_or_explicitly_uses_fixed_parameters():
    config = GridSearchConfig({"ridge": {"alpha": [10.]}}, min_train_periods=50)
    with pytest.raises(ValueError, match="only 0 available"):
        run(panel(), config)
    fallback = run(panel(), replace(config, on_insufficient_history="use_fixed"), parameters={"ridge": {"alpha": 2.}})
    fixed = run(panel(), None, parameters={"ridge": {"alpha": 2.}})
    pd.testing.assert_frame_equal(fallback.predictions, fixed.predictions)
    assert fallback.search_results.iloc[0].status == "insufficient_history"
    assert fallback.search_results.iloc[0].parameters == {"alpha": 2.}
    assert fallback.search_audit.empty and fixed.search_results.empty
    partial = replace(config, min_train_periods=8, n_splits=3)
    with pytest.raises(ValueError, match="only 2 available"):
        run(panel(), partial)


def test_transform_scoring_uses_original_units_and_custom_models():
    registry = ModelRegistry()
    registry.register("constant", lambda p: DummyRegressor(strategy="constant", **p))
    frame = panel()
    frame["target"] = 10.
    config = GridSearchConfig({"constant": {"constant": [0., np.log(10.)]}}, 3, scoring="mse")
    result = run(frame, config, models=("ols", "constant"), registry=registry,
                 target_transform=np.log, inverse_target_transform=np.exp)
    assert np.allclose(result.search_results.score, [81., 0.])
    assert result.search_results.iloc[1].selected
    assert np.allclose(result.predictions.prediction_constant, 10.)


@pytest.mark.parametrize("model,grid", [
    ("ridge", {"alpha": [0.1, 1.]}),
    ("elastic_net", {"alpha": [0.01], "l1_ratio": [0.2, 0.8]}),
    ("random_forest", {"n_estimators": [3], "max_depth": [2, 3]}),
    ("extra_trees", {"n_estimators": [3], "min_samples_leaf": [1, 2]}),
    ("hist_gradient_boosting", {"max_iter": [2], "max_leaf_nodes": [2, 3]}),
])
def test_shared_search_supports_all_model_families(model, grid):
    result = run(panel(), GridSearchConfig({model: grid}, 3, n_splits=1), models=("ols", model))
    assert len(result.search_results) == 2
    assert result.search_results.selected.sum() == 1
    assert np.isfinite(result.predictions[f"prediction_{model}"]).all()


def test_bad_config_and_candidate_errors_are_explicit():
    for kwargs in ({"scoring": "rmse"}, {"n_splits": 0}, {"n_splits": 1.5},
                   {"on_insufficient_history": "ignore"}, {"min_train_periods": True}):
        with pytest.raises(ValueError):
            GridSearchConfig(**{"parameter_grids": {"ridge": {"alpha": [1]}}, "min_train_periods": 3, **kwargs})
    for grid in ({}, {"ridge": {}}, {"ridge": {"alpha": []}}, {"ridge": {"model__alpha": [1]}}):
        with pytest.raises((ValueError, TypeError)):
            GridSearchConfig(grid, 3)
    with pytest.raises(ValueError, match="not configured"):
        ForecasterConfig(models=("ols", "ridge"), grid_search=GridSearchConfig({"extra_trees": {"max_depth": [2]}}, 3))
    with pytest.raises(ValueError, match="Outer fold.*candidate 0"):
        run(panel(), GridSearchConfig({"ridge": {"typo": [1]}}, 3))


def test_legacy_experiments_still_load(tmp_path):
    store = ExperimentStore(tmp_path)
    saved = store.save("legacy", run(panel(), None))
    manifest = json.loads((saved / "metadata.json").read_text())
    manifest["schema_version"] = 1
    (saved / "metadata.json").write_text(json.dumps(manifest))
    (saved / "search_results.json").unlink()
    (saved / "search_audit.json").unlink()
    loaded = store.load("legacy")
    assert loaded.search_results.empty and loaded.search_audit.empty
    store.report(tmp_path / "legacy.html")


def test_parallel_search_has_deterministic_selection_and_predictions():
    search = GridSearchConfig({
        "ridge": {"alpha": [0.1, 1.]},
        "extra_trees": {"max_depth": [2, 3]},
    }, 3, n_splits=2)
    kwargs = {"models": ("ridge", "extra_trees"), "parameters": {"extra_trees": {"n_estimators": 3}}}
    serial = run(panel(), search, **kwargs)
    parallel = run(panel(), search, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(serial.predictions, parallel.predictions)
    pd.testing.assert_frame_equal(
        serial.search_results.drop(columns="elapsed_seconds"),
        parallel.search_results.drop(columns="elapsed_seconds"),
    )


def test_equal_scores_select_first_candidate():
    frame = panel()
    frame["target"] = 10.
    result = run(frame, GridSearchConfig({"ridge": {"alpha": [100., 0.1]}}, 3))
    assert list(result.search_results.selected) == [True, False]
    assert result.fitted_models[(0, "ridge")].named_steps["model"].alpha == 100.
