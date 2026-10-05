# fx-ensemble

`fx-ensemble` is a Python framework for leakage-safe machine-learning ensembles on
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

| Name | Estimator | Model family | Strength / trade-off |
|---|---|---|---|
| `ols` | `LinearRegression` | linear regression | simple, fast, and easy to inspect; can be unstable with correlated predictors and only models linear effects |
| `ridge` | `Ridge` | regularized linear regression | stabilizes linear forecasts when predictors are correlated; shrinkage can add bias and does not select features |
| `elastic_net` | `ElasticNet` | sparse regularized linear regression | can shrink coefficients and remove weak predictors; requires tuning `alpha` and `l1_ratio`, and remains linear |
| `random_forest` | `RandomForestRegressor` | bagged decision trees | robust nonlinear baseline with low sensitivity to individual trees; can be slower and less interpretable |
| `extra_trees` | `ExtraTreesRegressor` | randomized bagged decision trees | randomized splits create a diverse nonlinear ensemble; the extra randomness can trade accuracy for variance reduction |
| `hist_gradient_boosting` | `HistGradientBoostingRegressor` | gradient-boosted decision trees | efficient nonlinear boosting model; more sensitive to hyperparameters and can overfit short training histories |

All built-in models use preprocessing fitted only on the training portion of each
validation fold: numeric imputation and missing-value indicators, categorical
imputation and encoding, plus scaling for linear models. The fitted transformations
are then applied unchanged to that fold's test period, preventing information from
future data from leaking into validation or out-of-sample forecasts. Relative model
performance is data-dependent, so compare candidates with the framework's temporal
validation or optional nested temporal grid search.

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

## Optional temporal grid search

A single grid-search engine supports every registered model, with a separate grid
for each model. Fixed parameters remain the default (`grid_search=None`). Configure
`tuning` and pass it to `ForecasterConfig`:

```python
from fxensemble import GridSearchConfig

tuning = GridSearchConfig(
    parameter_grids={
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "elastic_net": {"alpha": [0.001, 0.01], "l1_ratio": [0.2, 0.8]},
        "random_forest": {"max_depth": [4, 8], "min_samples_leaf": [2, 5]},
        "extra_trees": {"max_depth": [4, 8], "min_samples_leaf": [2, 5]},
        "hist_gradient_boosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 15]},
    },
    min_train_periods=12,
    n_splits=3,
    step_periods=1,
    scoring="mae",  # alternatively "mse"; positive loss, lower is better
    on_insufficient_history="raise",  # or explicitly "use_fixed"
)
forecaster = EnsembleForecaster(
    ForecasterConfig(
        models=tuple(tuning.parameter_grids),
        grid_search=tuning,
        model_parameters={
            "random_forest": {"n_estimators": 50},
            "extra_trees": {"n_estimators": 50},
        },
    ),
    ExpandingWindowConfig(min_train_periods=24),
)
# Call forecaster.fit_predict(...) with the same input contract as above.
```

At each outer forecast origin, tuning uses only the outer training sample. It
selects the latest `n_splits` eligible inner forecast dates after applying
`step_periods`, with expanding training windows and one validation date per split.
Inner training labels must be available by the inner origin; validation labels must
be available by the outer origin. Rows are split by date, never randomly shuffled.
Preprocessing is fitted afresh for each candidate and inner split. Declared feature
availability is checked at the inner origins as well as the outer origin.

The selected candidate minimizes the mean of the per-date MAE or MSE values. Each
validation date has equal weight, even if the number of entities differs. With target
transforms, training uses the transformed target and scoring uses predictions mapped
back to original target units. Transform functions must be stateless, pointwise
functions such as `np.log` and `np.exp`. Ties choose the first candidate in sklearn's
`ParameterGrid` order. The winning model is refitted on the complete outer training
sample before the untouched outer fold is predicted.

Grids use estimator parameter names (`alpha`, not `model__alpha`). Candidate values
override the corresponding `model_parameters`; other fixed parameters are retained.
Models without a grid keep their fixed parameters. Invalid candidates fail with
model, candidate, and fold context rather than being silently skipped. If fewer than
`n_splits` valid inner folds exist, the default is an error. With `use_fixed`, tuned
models use their fixed parameters for that outer fold and record the fallback.

`result.search_results` contains each candidate's parameters, per-split losses,
mean score, selected flag, status and evaluation time, by outer fold and model.
`result.search_audit` contains the inner training and validation date/availability
audits, shared across tuned models. Both are saved by `ExperimentStore` and shown in
HTML reports. Inner selection scores are separate from outer forecast metrics.

Keep grids small: the search costs `candidates × n_splits` fits per tuned model and
outer fold, plus the final refit. `n_jobs` still parallelizes models; inner candidate
fits are serial to avoid adding another parallelism layer.

Run a complete fixed-versus-tuned example into a new output directory:

```bash
python examples/grid_search.py --output /tmp/fx-grid-search
```

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
weights, fold audits, search results and inner search audits, and `report.html`. Forecaster configuration, feature lists,
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
