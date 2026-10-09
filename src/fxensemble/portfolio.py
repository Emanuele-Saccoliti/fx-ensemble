"""Risk-aware allocation and self-financing backtests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def volatility_scaled_scores(
    expected_returns: pd.Series,
    predicted_volatility: pd.Series,
    *,
    floor: float = 1e-8,
) -> pd.Series:
    """Scale expected returns by aligned, strictly positive volatility forecasts."""

    if floor <= 0:
        raise ValueError("floor must be positive")
    expected, volatility = expected_returns.align(predicted_volatility, join="inner")
    if expected.empty or expected.isna().any() or volatility.isna().any():
        raise ValueError("Expected returns and volatility must align without missing values")
    invalid_expected = np.count_nonzero(
        np.isnan(expected) | np.isinf(expected)
    )
    invalid_volatility = np.count_nonzero(
        np.isnan(volatility) | np.isinf(volatility)
    )
    if invalid_expected > 0:
        raise ValueError("Expected returns must be finite")
    if invalid_volatility > 0:
        raise ValueError("Predicted volatility must be finite")
    if (volatility <= 0).any():
        raise ValueError("Predicted volatility must be strictly positive")
    return expected / volatility.clip(lower=floor)


def rank_long_short_weights(
    scores: pd.Series,
    *,
    gross_exposure: float = 1.0,
    max_absolute_weight: float = 0.25,
) -> pd.Series:
    """Allocate equal-weight long and short books from cross-sectional ranks."""

    clean = scores.dropna()
    if len(clean) < 2 or gross_exposure <= 0 or max_absolute_weight <= 0:
        raise ValueError("Need at least two scores and positive exposure limits")
    side_count = len(clean) // 2
    side_exposure = gross_exposure / 2.0
    if side_count * max_absolute_weight + 1e-12 < side_exposure:
        raise ValueError("max_absolute_weight is infeasible for the requested gross exposure")
    order = clean.sort_values(kind="mergesort").index
    short_assets = order[:side_count]
    long_assets = order[-side_count:]
    weights = pd.Series(0.0, index=scores.index, dtype=float)
    weights.loc[short_assets] = -side_exposure / side_count
    weights.loc[long_assets] = side_exposure / side_count
    return weights


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate a positive-semidefinite covariance matrix with shrinkage."""

    if returns.shape[1] < 1:
        raise ValueError("At least one asset return column is required")
    complete = returns.astype(float).dropna(axis=0, how="any")
    if len(complete) < 2:
        raise ValueError("At least two complete observations are required")
    estimator = LedoitWolf().fit(complete.to_numpy())
    return pd.DataFrame(
        estimator.covariance_, index=complete.columns, columns=complete.columns
    )


def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_aversion: float = 10.0,
    gross_limit: float = 1.0,
    net_exposure: float = 0.0,
    max_absolute_weight: float = 0.25,
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
) -> pd.Series:
    """Solve a constrained mean-variance allocation with optional turnover penalty."""

    assets = expected_returns.index
    if len(assets) == 0 or set(covariance.index) != set(assets) or set(covariance.columns) != set(assets):
        raise ValueError("Expected returns and covariance assets must match")
    if risk_aversion <= 0 or gross_limit <= 0 or max_absolute_weight <= 0:
        raise ValueError("Risk aversion and exposure limits must be positive")
    if abs(net_exposure) > gross_limit + 1e-12:
        raise ValueError("Absolute net exposure cannot exceed the gross limit")
    if len(assets) * max_absolute_weight + 1e-12 < gross_limit:
        # A smaller realized gross is valid, so this only documents feasibility.
        gross_limit = len(assets) * max_absolute_weight

    mu = expected_returns.astype(float).to_numpy()
    sigma = covariance.loc[assets, assets].astype(float).to_numpy()
    sigma = (sigma + sigma.T) / 2.0
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Optimization inputs must be finite")
    previous = (
        previous_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        if previous_weights is not None
        else np.zeros(len(assets))
    )

    def objective(weights: np.ndarray) -> float:
        risk = 0.5 * risk_aversion * float(weights @ sigma @ weights)
        return_penalty = -float(mu @ weights)
        turnover = turnover_penalty * float(
            np.sum(np.sqrt(np.square(weights - previous) + 1e-12))
        )
        return risk + return_penalty + turnover

    start = np.repeat(net_exposure / len(assets), len(assets))
    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - net_exposure},
        {"type": "ineq", "fun": lambda weights: gross_limit - np.sum(np.abs(weights))},
    ]
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(-max_absolute_weight, max_absolute_weight)] * len(assets),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"Mean-variance optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if abs(weights.sum() - net_exposure) > 1e-7 or np.abs(weights).sum() > gross_limit + 1e-7:
        raise RuntimeError("Optimizer returned weights outside exposure constraints")
    return pd.Series(weights, index=assets, name="weight")


@dataclass(frozen=True)
class BacktestResult:
    weights: pd.DataFrame
    returns: pd.DataFrame


def continuous_backtest(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    date_column: str = "date",
    asset_column: str = "asset",
    weight_column: str = "target_weight",
    return_column: str = "asset_return",
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Backtest target weights with exact drift, turnover, and trading costs.

    Each date's return is earned after rebalancing at that date. Pre-trade weights
    therefore come from the prior post-cost holdings after their realized returns.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    weight_required = {date_column, asset_column, weight_column}
    return_required = {date_column, asset_column, return_column}
    if missing := weight_required.difference(target_weights.columns):
        raise KeyError(f"Missing weight columns: {sorted(missing)}")
    if missing := return_required.difference(asset_returns.columns):
        raise KeyError(f"Missing return columns: {sorted(missing)}")
    if target_weights.duplicated([date_column, asset_column]).any():
        raise ValueError("Target weights contain duplicate date-asset rows")
    if asset_returns.duplicated([date_column, asset_column]).any():
        raise ValueError("Asset returns contain duplicate date-asset rows")

    weights = target_weights.copy()
    returns = asset_returns.copy()
    weights[date_column] = pd.to_datetime(weights[date_column], errors="raise")
    returns[date_column] = pd.to_datetime(returns[date_column], errors="raise")
    if weights[date_column].isna().any():
        raise ValueError("Target weights contain missing dates")
    if returns[date_column].isna().any():
        raise ValueError("Asset returns contain missing dates")
    merged = weights.merge(
        returns,
        on=[date_column, asset_column],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Target weights and returns must have identical date-asset keys")
    merged = merged.drop(columns="_merge").sort_values([date_column, asset_column])
    if not np.isfinite(merged[[weight_column, return_column]].to_numpy(dtype=float)).all():
        raise ValueError("Weights and returns must be finite")

    cost_rate = cost_bps / 10_000.0
    weight_records: list[pd.DataFrame] = []
    return_records: list[dict[str, float | pd.Timestamp]] = []
    previous_assets: pd.Index | None = None
    previous_executed: np.ndarray | None = None
    previous_returns: np.ndarray | None = None
    previous_net_return: float | None = None

    for date, group in merged.groupby(date_column, sort=True):
        group = group.sort_values(asset_column)
        assets = pd.Index(group[asset_column])
        target = group[weight_column].to_numpy(dtype=float)
        period_returns = group[return_column].to_numpy(dtype=float)
        if previous_assets is None:
            pretrade = np.zeros(len(group))
        else:
            if not assets.equals(previous_assets):
                raise ValueError("The asset universe must remain constant across dates")
            denominator = 1.0 + float(previous_net_return)
            if denominator <= 0:
                raise RuntimeError("Portfolio wealth became non-positive")
            pretrade = previous_executed * (1.0 + previous_returns) / denominator

        transaction_cost = 0.0
        for _ in range(1_000):
            executed = (1.0 - transaction_cost) * target
            trade = executed - pretrade
            turnover = 0.5 * float(np.abs(trade).sum())
            updated_cost = cost_rate * turnover
            if abs(updated_cost - transaction_cost) <= 1e-14:
                transaction_cost = updated_cost
                break
            transaction_cost = updated_cost
        else:
            raise RuntimeError("Transaction-cost fixed point did not converge")
        executed = (1.0 - transaction_cost) * target
        trade = executed - pretrade
        turnover = 0.5 * float(np.abs(trade).sum())
        gross_return = float(target @ period_returns)
        net_return = float((1.0 - transaction_cost) * (1.0 + gross_return) - 1.0)

        detail = group[[date_column, asset_column]].copy()
        detail["target_weight"] = target
        detail["pretrade_weight"] = pretrade
        detail["executed_weight"] = executed
        detail["trade_weight"] = trade
        detail["turnover_contribution"] = 0.5 * np.abs(trade)
        weight_records.append(detail)
        return_records.append(
            {
                date_column: pd.Timestamp(date),
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "gross_exposure": float(np.abs(executed).sum()),
                "net_exposure": float(executed.sum()),
            }
        )
        previous_assets = assets
        previous_executed = executed
        previous_returns = period_returns
        previous_net_return = net_return

    return BacktestResult(
        weights=pd.concat(weight_records, ignore_index=True),
        returns=pd.DataFrame.from_records(return_records),
    )
