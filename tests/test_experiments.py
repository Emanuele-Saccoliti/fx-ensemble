import numpy as np
import pandas as pd
import pytest

from fxensemble import ExperimentStore, ForecastResult


def result(offset=0):
    return ForecastResult(
        predictions=pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3),
            "asset": ["EUR", "EUR", "EUR"],
            "actual": [1., 2., 3.],
            "prediction_ridge": np.array([1., 2., 3.]) + offset,
        }),
        ensemble_weights=pd.DataFrame(),
        fold_audit=pd.DataFrame({"forecast_date": pd.date_range("2020-01-01", periods=3)}),
        fitted_models={},
        metadata={"target_column": "actual", "forecaster": {"seed": 42}},
    )


def test_roundtrip_comparison_and_offline_report(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    original = result()
    saved = store.save("baseline", original, notes='<script>alert("x")</script>', metadata={"dataset": "v1"})
    store.save("alternative", result(1))
    loaded = store.load("baseline")
    pd.testing.assert_frame_equal(loaded.predictions, original.predictions)
    pd.testing.assert_frame_equal(loaded.fold_audit, original.fold_audit)
    assert loaded.metadata["configuration"] == original.metadata
    assert loaded.metadata["metadata"]["dataset"] == "v1"
    comparison = store.compare().set_index("experiment")
    assert comparison.loc["baseline", "rmse"] == 0
    assert comparison.loc["alternative", "rmse"] == 1
    assert (saved / "report.html").exists()
    report = store.report(tmp_path / "comparison.html").read_text()
    assert "baseline" in report and "alternative" in report
    assert "&lt;script&gt;" in report and "<script>" not in report
    assert "https://" not in report
    assert store.compare(["baseline"]).shape[0] == 1


def test_invalid_names_duplicates_and_empty_store(tmp_path):
    store = ExperimentStore(tmp_path)
    assert store.names() == []
    with pytest.raises(ValueError, match="No experiments"):
        store.compare()
    for name in ("../escape", "/absolute", ".hidden", ""):
        with pytest.raises(ValueError, match="names"):
            store.save(name, result())
    store.save("run", result())
    with pytest.raises(FileExistsError):
        store.save("run", result(2))
    assert store.load("run").metrics.iloc[0]["rmse"] == 0
    with pytest.raises(ValueError, match="unique"):
        store.compare(["run", "run"])


def test_failed_save_leaves_no_partial_run(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_json", fail)
    with pytest.raises(OSError, match="disk full"):
        store.save("broken", result())
    assert list(tmp_path.iterdir()) == []


def test_manual_result_requires_target_and_serializable_metadata(tmp_path):
    store = ExperimentStore(tmp_path)
    run = result()
    manual = ForecastResult(run.predictions, run.ensemble_weights, run.fold_audit, {})
    with pytest.raises(ValueError, match="actual_column"):
        store.save("manual", manual)
    store.save("manual", manual, actual_column="actual")
    with pytest.raises(TypeError, match="JSON serializable"):
        store.save("bad", run, metadata={"callable": lambda: None})
    assert store.names() == ["manual"]
