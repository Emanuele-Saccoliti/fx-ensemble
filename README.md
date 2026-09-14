# fxensemble

`fxensemble` is a Python framework for leakage-safe machine-learning ensembles on
time-indexed panel data.

The framework provides:

- expanding-window cross-validation based on the date each target became observable;
- fold-local imputation, missing-value indicators, scaling, and categorical encoding;
- deterministic linear, regularized, random-forest, extra-trees, and histogram-gradient-boosting models;
- a registry for custom scikit-learn-compatible regressors;
- equal-weight and past-OOS-performance ensemble rules;
- log-target support for positive quantities such as realized volatility;
- forecast metrics, volatility-scaled signals, covariance shrinkage, constrained allocation, and self-financing backtests;
- fold audits and ensemble-weight artifacts for reproducibility;
- model-level parallel execution through `n_jobs`.

## Install

From this directory:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Minimal volatility example

```python
import numpy as np

from fxensemble import (
    EnsembleForecaster,
    ExpandingWindowConfig,
    ForecasterConfig,
    PerformanceEnsembleConfig,
)

forecaster = EnsembleForecaster(
    ForecasterConfig(
        models=("ridge", "extra_trees", "hist_gradient_boosting"),
        n_jobs=3,
        model_parameters={"extra_trees": {"n_estimators": 300}},
        performance_ensemble=PerformanceEnsembleConfig(
            minimum_history_periods=12
        ),
    ),
    ExpandingWindowConfig(min_train_periods=60),
)

result = forecaster.fit_predict(
    panel,
    numeric_features=["rate_diff", "inflation_diff", "lagged_volatility"],
    categorical_features=["currency"],
    target_column="realized_volatility_next_month",
    date_column="date",
    label_available_column="target_release_date",
    id_columns=["currency"],
    feature_availability={
        "rate_diff": "macro_release_date",
        "inflation_diff": "macro_release_date",
        "lagged_volatility": "date",
    },
    target_transform=np.log,
    inverse_target_transform=np.exp,
)

result.predictions.to_parquet("volatility_forecasts.parquet")
result.fold_audit.to_csv("fold_audit.csv", index=False)
```

Every input row is one entity at one forecast date. `label_available_column` is the
date on which that row's target was known. For each forecast date, the training set
contains only earlier rows whose labels were already known. Performance-based
ensemble weights obey the same rule when they consume historical forecast errors.

The caller remains responsible for supplying vintage-correct features. When release
date columns are available, pass `feature_availability={feature: release_column}` and
the framework will reject values dated after the forecast origin.

## Built-in models

| Name | Estimator | Fold-local treatment |
|---|---|---|
| `ols` | `LinearRegression` | imputation, indicators, scaling, encoding |
| `ridge` | `Ridge` | imputation, indicators, scaling, encoding |
| `elastic_net` | `ElasticNet` | imputation, indicators, scaling, encoding |
| `random_forest` | `RandomForestRegressor` | imputation, indicators, encoding |
| `extra_trees` | `ExtraTreesRegressor` | imputation, indicators, encoding |
| `hist_gradient_boosting` | `HistGradientBoostingRegressor` | imputation, indicators, encoding |

Register any compatible regressor without modifying the framework:

```python
from sklearn.dummy import DummyRegressor
from fxensemble import ModelRegistry

registry = ModelRegistry(seed=7)
registry.register("baseline", lambda params: DummyRegressor(**params))
```

Pass the registry to `EnsembleForecaster(..., registry=registry)` and include
`"baseline"` in `ForecasterConfig.models`.

See [`docs/architecture.md`](docs/architecture.md) for data contracts and component
boundaries. The runnable example is in
[`examples/volatility_forecast.py`](examples/volatility_forecast.py).

## Save and compare experiments

`ExperimentStore` saves named forecasting runs locally and generates standalone HTML
reports without extra dependencies or external services:

```python
from fxensemble import ExperimentStore

store = ExperimentStore("experiments")
store.save("baseline", result, notes="Initial configuration", metadata={"dataset": "panel-v1"})
# After running a different configuration on the same panel:
store.save("alternative", alternative_result, metadata={"dataset": "panel-v1"})
print(store.compare())
store.report("experiments/comparison.html")
restored = store.load("baseline")
```

Each run contains `metadata.json`, JSON tables for predictions, metrics, ensemble
weights and fold audits, and `report.html`. Forecaster configuration, feature lists,
column roles and transform names are captured automatically. Add dataset versions
and custom model details through `metadata`. Names are unique and existing runs
cannot be overwritten. Writes are staged so failed saves do not leave partial runs.

`compare(names=[...])` and `report(..., names=[...])` select a subset; by default they
include all saved runs. Metrics use each run's own sample, so compare matching targets,
dates and entities. The report shows configuration details and fold audits, with
previews limited to 100 predictions and weight rows; saved tables retain all rows.
Fitted estimators and executable transforms are not serialized, and loading a run
restores analysis artifacts rather than a runnable model. Callable names document
transforms but cannot reproduce their implementations. Metadata must be JSON serializable.

Run the complete two-experiment example into a new output directory:

```bash
python examples/compare_experiments.py --output /tmp/fx-experiments
```
