"""Compare fixed parameters and nested temporal tuning on the same panel.

Usage: python examples/grid_search.py --output /tmp/fx-grid-search
"""

import argparse
from pathlib import Path

from volatility_forecast import make_panel
from fxensemble import (
    EnsembleForecaster, ExperimentStore, ExpandingWindowConfig,
    ForecasterConfig, GridSearchConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = ExperimentStore(args.output)
    panel = make_panel()
    search = GridSearchConfig(
        parameter_grids={
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "extra_trees": {"max_depth": [3], "min_samples_leaf": [1, 5]},
        },
        min_train_periods=12,
        n_splits=3,
        scoring="mae",
    )
    for name, tuning in (("fixed", None), ("grid-search", search)):
        forecaster = EnsembleForecaster(
            ForecasterConfig(
                models=("ridge", "extra_trees"), seed=5, n_jobs=2,
                model_parameters={"extra_trees": {"n_estimators": 10}},
                grid_search=tuning,
            ),
            ExpandingWindowConfig(min_train_periods=18),
        )
        result = forecaster.fit_predict(
            panel, numeric_features=["macro", "lagged_volatility"],
            categorical_features=["currency"], id_columns=["currency"],
            target_column="realized_volatility",
        )
        store.save(name, result, metadata={"dataset": "synthetic-volatility", "dataset_seed": 12})
    print(store.compare().to_string(index=False))
    print(f"Report: {store.report(args.output / 'comparison.html').resolve()}")


if __name__ == "__main__":
    main()
