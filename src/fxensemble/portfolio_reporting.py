"""Validation, summaries and offline SVG charts for saved portfolio backtests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from fxensemble.metrics import performance_metrics
from fxensemble.portfolio import BacktestResult


def prepare_backtest(
    backtest: BacktestResult,
    date_column: str,
    periods_per_year: int,
    initial_capital: float,
) -> BacktestResult:
    """Copy and sort a periodic backtest without dropping any observations."""
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral):
        raise TypeError("periods_per_year must be a positive integer")
    if periods_per_year < 1 or not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Period frequency and initial capital must be finite and positive")
    returns = _dated_table(backtest.returns, date_column)
    weights = _dated_table(backtest.weights, date_column)
    if returns[date_column].duplicated().any():
        raise ValueError("Portfolio returns must contain one row per date")
    if set(weights[date_column]) != set(returns[date_column]):
        raise ValueError("Portfolio weights and returns must cover the same dates")
    required = {"gross_return", "net_return", "turnover", "transaction_cost"}
    if missing := required.difference(returns.columns):
        raise ValueError(f"Missing portfolio return columns: {sorted(missing)}")
    returns[list(required)] = returns[list(required)].astype(float)
    invalid_count = np.count_nonzero(~np.isfinite(returns[list(required)].to_numpy()))
    if invalid_count > 0:
        raise ValueError("Portfolio returns, turnover and costs must be finite")
    if (returns[["gross_return", "net_return"]] < -1).any().any():
        raise ValueError("Portfolio reports require returns greater than or equal to -100%")
    if (returns[["turnover", "transaction_cost"]] < 0).any().any():
        raise ValueError("Portfolio turnover and transaction costs cannot be negative")
    return BacktestResult(weights=weights, returns=returns)


def _dated_table(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if frame.empty or date_column not in frame:
        raise ValueError(f"Portfolio tables must be non-empty and contain {date_column!r}")
    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="raise")
    if ordered[date_column].isna().any():
        raise ValueError("Portfolio dates cannot be missing")
    return ordered.sort_values(date_column, kind="stable").reset_index(drop=True)


def _wealth(returns: pd.Series, initial_capital: float) -> np.ndarray:
    wealth = initial_capital * np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    if np.count_nonzero(~np.isfinite(wealth)) > 0:
        raise ValueError("Portfolio compounded capital must be finite")
    return wealth


def portfolio_metrics(
    backtest: BacktestResult, periods_per_year: int, initial_capital: float
) -> pd.DataFrame:
    """Summarize gross and net periodic returns, including the initial capital."""
    records = []
    for basis in ("gross", "net"):
        returns = backtest.returns[f"{basis}_return"]
        wealth = _wealth(returns, initial_capital)
        metrics = performance_metrics(returns, periods_per_year=periods_per_year)
        records.append(
            {
                "basis": basis,
                "n_periods": metrics["n_periods"],
                "initial_capital": initial_capital,
                "final_capital": float(wealth[-1]),
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "maximum_drawdown": metrics["maximum_drawdown"],
            }
        )
    summary = pd.DataFrame.from_records(records)
    if np.count_nonzero(~np.isfinite(summary.select_dtypes("number").to_numpy())) > 0:
        raise ValueError("Portfolio summary metrics must be finite")
    return summary


def _summary_table(metrics: pd.DataFrame) -> str:
    labels = {
        "basis": "Returns",
        "n_periods": "Periods",
        "initial_capital": "Initial capital",
        "final_capital": "Final capital",
        "total_return": "Total return",
        "annualized_return": "Annualized return",
        "annualized_volatility": "Annualized volatility",
        "maximum_drawdown": "Maximum drawdown",
    }
    formatted = metrics.copy()
    formatted["basis"] = formatted["basis"].map({"gross": "Gross", "net": "Net of costs"})
    for column in (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    ):
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    for column in ("initial_capital", "final_capital"):
        formatted[column] = formatted[column].map(lambda value: f"{value:,.2f}")
    return (
        '<div class="table">'
        + formatted.rename(columns=labels).to_html(index=False, escape=True)
        + "</div>"
    )


def _line_chart(
    title: str, labels: Sequence[str], series: Mapping[str, np.ndarray], *, percent: bool = False
) -> str:
    """Render all observations in a responsive SVG, with no remote resources."""
    colors = ("#147d92", "#bd572c")
    values = np.concatenate(list(series.values()))
    low, high = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    span = high - low or 1.0
    low, high = low - span * 0.05, high + span * 0.05
    parts = [
        f'<figure class="portfolio-chart"><figcaption>{escape(title)}</figcaption>',
        f'<svg viewBox="0 0 720 270" role="img" aria-label="{escape(title, quote=True)}">',
        f"<title>{escape(title)}</title>",
        _chart_axes(labels, low, high, percent),
    ]
    legend = []
    for (name, data), color in zip(series.items(), colors):
        xs = np.linspace(78, 690, len(data))
        ys = 218 - (data - low) / (high - low) * 194
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        if len(data) == 1:
            parts.append(f'<circle cx="{xs[0]:.2f}" cy="{ys[0]:.2f}" r="3" fill="{color}"/>')
        legend.append(
            f'<span><i class="chart-key" style="background:{color}"></i>{escape(name)}</span>'
        )
    parts.extend(['</svg><div class="chart-legend">', "".join(legend), "</div></figure>"])
    return "".join(parts)


def _chart_axes(labels: Sequence[str], low: float, high: float, percent: bool) -> str:
    parts = []
    for tick in np.linspace(low, high, 5):
        y = 218 - (tick - low) / (high - low) * 194
        label = f"{tick:.1%}" if percent else f"{tick:,.3g}"
        parts.append(f'<path d="M78 {y:.2f} H690" stroke="#e1e7ed"/>')
        parts.append(
            f'<text x="68" y="{y + 4:.2f}" text-anchor="end" font-size="12">{label}</text>'
        )
    ticks = sorted({0, (len(labels) - 1) // 2, len(labels) - 1})
    for index in ticks:
        x = 78 + 612 * index / max(len(labels) - 1, 1)
        anchor = "start" if index == 0 else "end" if index == len(labels) - 1 else "middle"
        parts.append(
            f'<text x="{x:.2f}" y="246" text-anchor="{anchor}" font-size="12">'
            f"{escape(labels[index])}</text>"
        )
    return "".join(parts)


def _portfolio_charts(backtest: BacktestResult, date_column: str, capital: float) -> str:
    returns = backtest.returns
    dates = returns[date_column].dt.strftime("%Y-%m-%d").tolist()
    wealth = {
        name: _wealth(returns[f"{basis}_return"], capital)
        for name, basis in (("Gross", "gross"), ("Net of costs", "net"))
    }
    drawdowns = {
        name: values / np.maximum.accumulate(values) - 1 for name, values in wealth.items()
    }
    return (
        '<div class="portfolio-charts">'
        + "".join(
            [
                _line_chart("Portfolio capital", ["Start", *dates], wealth),
                _line_chart(
                    "Drawdown from previous peak", ["Start", *dates], drawdowns, percent=True
                ),
                _line_chart(
                    "Turnover per period",
                    dates,
                    {"Half of absolute traded weight": returns["turnover"].to_numpy()},
                    percent=True,
                ),
                _line_chart(
                    "Trading costs per period (bps)",
                    dates,
                    {"Cost / opening capital": returns["transaction_cost"].to_numpy() * 10_000},
                ),
            ]
        )
        + "</div>"
    )


def portfolio_report(
    backtest: BacktestResult, metrics: pd.DataFrame, config: Mapping[str, Any]
) -> str:
    """Render portfolio metrics, charts and bounded previews of the full ledgers."""
    capital = float(config["initial_capital"])
    date_column = str(config["date_column"])
    dates = backtest.returns[date_column]
    net_wealth = _wealth(backtest.returns["net_return"], capital)
    cost_amounts = net_wealth[:-1] * backtest.returns["transaction_cost"].to_numpy()
    summary = (
        f"<h3>Portfolio backtest</h3><p>{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d} · "
        f"{len(dates)} periods · {int(config['periods_per_year'])} periods per year · "
        f"initial capital {capital:,.2f} units.</p>"
        "<p>Each date identifies a return period; capital is shown after that period. "
        "Start is the capital before the first period. Observations are equally spaced; "
        "annualization uses the declared frequency. Gross capital compounds target-weight returns "
        "before trading costs; net capital compounds the supplied net returns.</p>"
        f"<p>Total trading costs paid: {cost_amounts.sum():,.2f} capital units. "
        "This sum excludes the subsequent returns forgone on those costs. "
        f"Mean turnover per period: {backtest.returns['turnover'].mean():.2%}.</p>"
    )
    parts = [summary, _summary_table(metrics), _portfolio_charts(backtest, date_column, capital)]
    for title, frame in (
        ("Portfolio returns", backtest.returns),
        ("Portfolio weights", backtest.weights),
    ):
        parts.extend(
            [
                f"<details><summary>{title} (first 100 rows)</summary>",
                (
                    "<p>Full tables are saved with the experiment. Returns, weights, turnover "
                    'and transaction costs below are fractions, not percentages.</p><div class="table">'
                ),
                frame.head(100).to_html(
                    index=False, escape=True, float_format=lambda x: f"{x:.6g}"
                ),
                "</div></details>",
            ]
        )
    return "".join(parts)
