"""Local experiment artifacts and standalone HTML comparison reports."""

from __future__ import annotations

import json
import platform
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxensemble.forecaster import ForecastResult
from fxensemble.metrics import regression_metrics
from fxensemble.portfolio import BacktestResult
from fxensemble.portfolio_reporting import (
    portfolio_metrics,
    portfolio_report,
    prepare_backtest,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"Experiment metadata must be JSON serializable: {type(value).__name__}")


@dataclass(frozen=True)
class Experiment:
    """Reloaded artifacts; fitted estimators are deliberately not serialized."""

    name: str
    metadata: Mapping[str, Any]
    predictions: pd.DataFrame
    ensemble_weights: pd.DataFrame
    fold_audit: pd.DataFrame
    metrics: pd.DataFrame
    search_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: BacktestResult | None = None
    backtest_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


class ExperimentStore:
    """Save named runs without overwriting them, and compare their saved metrics."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
            raise ValueError("Experiment names must use 1-100 letters, digits, dots, underscores or hyphens")
        return self.directory / name

    def save(
        self,
        name: str,
        result: ForecastResult,
        *,
        actual_column: str | None = None,
        prediction_columns: Sequence[str] | None = None,
        notes: str = "",
        metadata: Mapping[str, Any] | None = None,
        backtest: BacktestResult | None = None,
        backtest_date_column: str = "date",
        periods_per_year: int = 12,
        initial_capital: float = 100.0,
    ) -> Path:
        """Save a run and its HTML report. Supply dataset/version details in metadata.

        Configuration is captured automatically for results from EnsembleForecaster.
        Explicit actual/prediction columns support manually constructed results.
        An optional backtest is supplied by the caller, never inferred from forecasts.
        Its period frequency and initial capital are recorded for reporting.
        """
        destination = self._path(name)
        if destination.exists():
            raise FileExistsError(f"Experiment already exists: {name}")
        actual = actual_column or result.metadata.get("target_column")
        if not actual:
            raise ValueError("actual_column is required for results without target metadata")
        columns = list(prediction_columns) if prediction_columns is not None else [
            column for column in result.predictions if column.startswith("prediction_")
        ]
        if not columns:
            raise ValueError("At least one prediction column is required")
        metrics = regression_metrics(result.predictions, actual_column=actual, prediction_columns=columns)
        backtest_summary = pd.DataFrame()
        backtest_config = None
        if backtest is not None:
            backtest = prepare_backtest(
                backtest, backtest_date_column, periods_per_year, initial_capital
            )
            backtest_summary = portfolio_metrics(backtest, periods_per_year, initial_capital)
            backtest_config = {
                "date_column": backtest_date_column,
                "periods_per_year": int(periods_per_year),
                "initial_capital": float(initial_capital),
            }
        manifest = {
            "schema_version": 3,
            "environment": {
                "python": platform.python_version(),
                **{package: version(package) for package in ("numpy", "pandas", "scikit-learn", "scipy")},
            },
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "configuration": dict(result.metadata),
            "metadata": dict(metadata or {}),
            "actual_column": actual,
            "prediction_columns": columns,
            "backtest": backtest_config,
        }
        manifest_text = json.dumps(manifest, indent=2, default=_json_default, allow_nan=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
        try:
            (staging / "metadata.json").write_text(manifest_text, encoding="utf-8")
            tables = {
                "predictions": result.predictions,
                "ensemble_weights": result.ensemble_weights,
                "fold_audit": result.fold_audit,
                "metrics": metrics,
                "search_results": result.search_results,
                "search_audit": result.search_audit,
            }
            if backtest is not None:
                tables.update({
                    "portfolio_weights": backtest.weights,
                    "portfolio_returns": backtest.returns,
                    "portfolio_metrics": backtest_summary,
                })
            for key, frame in tables.items():
                frame.to_json(staging / f"{key}.json", orient="table", date_format="iso", index=False, double_precision=15)
            experiment = Experiment(
                name, manifest, result.predictions, result.ensemble_weights,
                result.fold_audit, metrics, result.search_results, result.search_audit,
                backtest, backtest_summary,
            )
            (staging / "report.html").write_text(_report([experiment]), encoding="utf-8")
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination

    def load(self, name: str) -> Experiment:
        """Load portable JSON tables, without executing serialized Python code."""
        path = self._path(name)
        manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in (1, 2, 3):
            raise ValueError("Unsupported experiment schema version")
        search_tables = {}
        if manifest["schema_version"] >= 2:
            search_tables = {
                key: pd.read_json(path / f"{key}.json", orient="table")
                for key in ("search_results", "search_audit")
            }
        backtest = None
        backtest_summary = pd.DataFrame()
        if manifest["schema_version"] == 3 and manifest.get("backtest") is not None:
            backtest = BacktestResult(
                weights=pd.read_json(path / "portfolio_weights.json", orient="table"),
                returns=pd.read_json(path / "portfolio_returns.json", orient="table"),
            )
            backtest_summary = pd.read_json(path / "portfolio_metrics.json", orient="table")
        return Experiment(name=name, metadata=manifest, backtest=backtest,
                          backtest_metrics=backtest_summary, **search_tables, **{
            key: pd.read_json(path / f"{key}.json", orient="table")
            for key in ("predictions", "ensemble_weights", "fold_audit", "metrics")
        })

    def names(self) -> list[str]:
        """List completed experiments in name order."""
        if not self.directory.exists():
            return []
        return sorted(path.name for path in self.directory.iterdir()
                      if path.is_dir() and not path.name.startswith(".")
                      and (path / "metadata.json").is_file())

    def _experiments(self, names: Sequence[str] | None) -> list[Experiment]:
        selected = self.names() if names is None else list(names)
        if not selected:
            raise ValueError("No experiments selected")
        if len(set(selected)) != len(selected):
            raise ValueError("Experiment names must be unique")
        return [self.load(name) for name in selected]

    def compare(self, names: Sequence[str] | None = None) -> pd.DataFrame:
        """Return metrics on each run's own sample, without ranking different samples."""
        return _comparison(self._experiments(names))

    def report(self, output: str | Path, names: Sequence[str] | None = None) -> Path:
        """Write an offline HTML comparison and individual run details."""
        html = _report(self._experiments(names))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        return output


def _comparison(experiments: Sequence[Experiment]) -> pd.DataFrame:
    frames = []
    for experiment in experiments:
        date_column = experiment.metadata["configuration"].get("date_column")
        dates = experiment.predictions.get(date_column)
        frames.append(experiment.metrics.assign(
            experiment=experiment.name,
            target=experiment.metadata["actual_column"],
            forecast_start=None if dates is None else dates.min(),
            forecast_end=None if dates is None else dates.max(),
        ))
    combined = pd.concat(frames, ignore_index=True)
    leading = ["experiment", "target", "forecast_start", "forecast_end", "model", "n"]
    return combined[leading + [column for column in combined if column not in leading]]


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No observations.</p>"
    return '<div class="table">' + frame.to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}") + "</div>"


def _report(experiments: Sequence[Experiment]) -> str:
    parts = ["<h1>Forecast experiments</h1>",
             ("<p>Metrics use each experiment's own evaluation sample. Compare runs with the same "
              "target, dates and entities; sample sizes alone do not establish comparability.</p>"),
             "<h2>Metric comparison</h2>", _table(_comparison(experiments))]
    for experiment in experiments:
        parts.extend([
            f"<h2>{escape(experiment.name)}</h2>",
            f"<p>{escape(experiment.metadata['notes'])}</p>",
            f"<p>{len(experiment.predictions)} forecast rows · {len(experiment.fold_audit)} folds</p>",
            "<details><summary>Configuration and metadata</summary><pre>"
            + escape(json.dumps(experiment.metadata, indent=2, default=_json_default)) + "</pre></details>",
            "<h3>Metrics</h3>", _table(experiment.metrics),
            "<details><summary>Fold audit</summary>", _table(experiment.fold_audit), "</details>",
            "<details><summary>Ensemble weights (first 100 rows)</summary>",
            _table(experiment.ensemble_weights.head(100)), "</details>",
            "<details><summary>Predictions (first 100 rows)</summary>",
            _table(experiment.predictions.head(100)), "</details>",
        ])
        if experiment.backtest is not None:
            parts.append(portfolio_report(
                experiment.backtest, experiment.backtest_metrics,
                experiment.metadata["backtest"],
            ))
        if not experiment.search_results.empty:
            selected = experiment.search_results.loc[experiment.search_results["selected"]]
            parts.extend([
                "<h3>Temporal grid search</h3>",
                ("<p>Scores are mean inner-fold losses on original target units (lower is better). "
                "They are selection scores, not outer backtest performance. "
                "Insufficient-history rows record use of fixed parameters.</p>"),
                "<details open><summary>Selected parameters (first 100 rows)</summary>",
                _table(selected.head(100)), "</details>",
                "<details><summary>All candidates (first 100 rows)</summary>",
                _table(experiment.search_results.head(100)), "</details>",
                "<details><summary>Inner fold audit (first 100 rows)</summary>",
                _table(experiment.search_audit.head(100)), "</details>",
            ])
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecast experiments</title><style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#182738;background:#f5f7fa}
h1,h2,h3{color:#123b55}h2{margin-top:36px}p{line-height:1.6}.table{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px 14px;border:1px solid #dce3ea;text-align:right}
th{background:#e7eff5}summary{cursor:pointer;padding:12px 0;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}
.portfolio-charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:20px}
.portfolio-chart{margin:0;background:white;padding:16px;border:1px solid #dce3ea;border-radius:8px}
.portfolio-chart figcaption{font-weight:600;margin-bottom:12px}.portfolio-chart svg{width:100%;height:auto;display:block}
.chart-legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px}.chart-key{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:6px}
</style></head><body>""" + "".join(parts) + "</body></html>"
