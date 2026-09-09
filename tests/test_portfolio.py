import numpy as np
import pandas as pd

from fxensemble import (
    continuous_backtest,
    ledoit_wolf_covariance,
    mean_variance_weights,
    rank_long_short_weights,
    volatility_scaled_scores,
)


def test_risk_scaled_rank_portfolio_respects_exposure_constraints():
    expected = pd.Series({"A": 0.04, "B": 0.01, "C": -0.01, "D": -0.03})
    volatility = pd.Series({"A": 0.2, "B": 0.1, "C": 0.1, "D": 0.3})
    scores = volatility_scaled_scores(expected, volatility)
    weights = rank_long_short_weights(scores, gross_exposure=1.0, max_absolute_weight=0.25)
    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(weights.abs().sum(), 1.0)
    assert weights.abs().max() <= 0.25


def test_shrinkage_covariance_and_optimizer_are_well_formed():
    rng = np.random.default_rng(4)
    sample = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("ABCD"))
    covariance = ledoit_wolf_covariance(sample)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    weights = mean_variance_weights(
        pd.Series([0.03, 0.01, -0.01, -0.02], index=list("ABCD")),
        covariance,
        gross_limit=1.0,
        max_absolute_weight=0.4,
    )
    assert abs(weights.sum()) < 1e-7
    assert weights.abs().sum() <= 1.0 + 1e-7
    assert weights.abs().max() <= 0.4 + 1e-7


def test_continuous_backtest_reconciles_costs_and_drift_exactly():
    dates = pd.to_datetime(["2022-01-01", "2022-02-01"])
    target = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B"] * 2,
            "target_weight": [0.5, -0.5, 0.5, -0.5],
        }
    )
    returns = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B"] * 2,
            "asset_return": [0.10, -0.05, 0.02, 0.01],
        }
    )
    result = continuous_backtest(target, returns, cost_bps=100.0)
    first_return = result.returns.iloc[0]
    assert np.isclose(first_return["transaction_cost"], 0.01 * first_return["turnover"])
    assert np.isclose(
        first_return["net_return"],
        (1.0 - first_return["transaction_cost"])
        * (1.0 + first_return["gross_return"])
        - 1.0,
    )
    first_weights = result.weights[result.weights["date"] == dates[0]]
    second_weights = result.weights[result.weights["date"] == dates[1]]
    expected_pretrade = (
        first_weights["executed_weight"].to_numpy()
        * (1.0 + returns.loc[returns["date"] == dates[0], "asset_return"].to_numpy())
        / (1.0 + first_return["net_return"])
    )
    assert np.allclose(second_weights["pretrade_weight"], expected_pretrade)
