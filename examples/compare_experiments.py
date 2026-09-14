"""Run two configurations and save an offline comparison report.

Usage: python examples/compare_experiments.py --output /tmp/fx-experiments
"""

import argparse
from pathlib import Path

from volatility_forecast import make_panel
from fxensemble import EnsembleForecaster, ExperimentStore, ExpandingWindowConfig, ForecasterConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = ExperimentStore(args.output)
    panel = make_panel()
    for name, alpha in (("ridge-alpha-1", 1.0), ("ridge-alpha-10", 10.0)):
        forecaster = EnsembleForecaster(
            ForecasterConfig(models=("ols", "ridge"), model_parameters={"ridge": {"alpha": alpha}}),
            ExpandingWindowConfig(min_train_periods=18),
        )
        result = forecaster.fit_predict(
            panel,
            numeric_features=["macro", "lagged_volatility"],
            categorical_features=["currency"],
            id_columns=["currency"],
            target_column="realized_volatility",
        )
        store.save(name, result, metadata={"dataset": "synthetic-volatility", "dataset_seed": 12})
    print(store.compare().to_string(index=False))
    print(f"Report: {store.report(args.output / 'comparison.html').resolve()}")


if __name__ == "__main__":
    main()
