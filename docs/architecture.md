# Architecture and scientific contracts

## Pipeline

```mermaid
flowchart LR
    D[Point-in-time panel] --> S[ExpandingWindowSplitter]
    S --> P[Fold-local preprocessing]
    P --> M[Registered regressors]
    M --> O[Out-of-sample predictions]
    O --> E[Point-in-time ensembles]
    E --> R[Risk-aware allocation]
    R --> B[Continuous self-financing backtest]
```

`EnsembleForecaster` orchestrates the first five stages. Allocation and backtesting
are separate functions so a forecast experiment cannot silently change the economic
evaluation protocol.

## Input contract

The forecaster consumes a pandas `DataFrame`. Multiple entities may share a date.
The required semantic fields are:

| Field | Meaning |
|---|---|
| forecast date | Information set and OOS prediction origin |
| target | Value being forecast and later scored |
| label-available date | First date on which that target can enter training or ensemble error history |
| numeric/categorical features | Predictors as visible in the forecast-date vintage |
| optional feature-available dates | Audit dates used to reject future feature values |

The splitter requires `training date < forecast date` and
`label-available date <= forecast date`. A row with an unavailable label can still
be forecast and retained for later evaluation, but it cannot enter model fitting or
adaptive ensemble weights prematurely.

## Fold isolation

Each estimator receives a fresh scikit-learn pipeline in every fold. Numeric medians
or means, missing-value indicator selection, standardization, and categorical levels
are learned on that fold's training rows only. Linear models are standardized by
default. Tree models use unscaled numeric features. Unknown test categories are
ignored by the encoder.

`ForecastResult.fold_audit` records the train range, row and period counts, latest
label date, test size, and model set. Fitted fold models are omitted by default to
keep large runs memory-efficient; set `store_fitted_models=True` when model inspection
is required.

## Ensemble rules

The equal ensemble averages all configured forecasts. The performance ensemble uses
inverse trailing MSE or MAE. At forecast date `t`, its error history includes only
predictions made before `t` whose realized labels are available by `t`. It uses equal
weights during the configured warm-up. Weights are normalized to one and returned as
a separate dated artifact.

## Scaling and extension

Panel rows, feature sets, and model sets are unrestricted by the API. Models within a
fold can execute concurrently through `ForecasterConfig.n_jobs`. Built-in tree models
default to one internal worker, preventing accidental nested parallelism. Custom model
factories can expose their own compute strategy through `model_parameters`.

For very large data, keep the panel in a columnar store, load the required sample once,
and leave `store_fitted_models=False`. Fold outputs contain only identifiers, targets,
dates, and predictions. The design deliberately avoids dependencies on a particular
FX vendor, macroeconomic source, project directory, or experiment configuration.

## Portfolio accounting

`continuous_backtest` treats target weights as weights after rebalancing and before
the period return. It solves transaction cost and executed holdings as a fixed point,
then drifts those holdings through asset returns to produce the next date's pre-trade
weights. Reported net returns satisfy

```text
1 + net_return = (1 - transaction_cost) * (1 + target_weight @ asset_return)
```

Turnover is one half of absolute traded weight. This convention is explicit and is
tested through the returned holdings ledger.

## Experiment artifacts

`ForecastResult.metadata` captures forecasting configuration and input column roles.
`ExperimentStore` persists a named result through a staged directory, then exposes
loading, metric comparison and offline HTML reporting. JSON tables preserve dates
and table schemas without pickling estimators. Floating-point values are serialized
with 15 decimal places; this is an analysis archive, not a bit-exact model checkpoint.
Custom metadata records dataset identifiers or other provenance supplied by callers.
Reports compare saved metrics on each run's original sample and do not align samples
or claim statistical superiority. Prediction and weight previews are bounded to keep
HTML output manageable; the complete tables remain in the saved artifacts.

## Nested temporal hyperparameter selection

`ForecasterConfig.grid_search` optionally supplies a `GridSearchConfig` with
per-estimator grids. `EnsembleForecaster` constructs one shared set of inner folds
from each outer training sample using `ExpandingWindowSplitter`. The latest
`n_splits` eligible dates after inner `step_periods` thinning are used. A split
contains all eligible rows at its validation date, preserving entity/date grouping.
`allow_empty=True` lets the inner splitter return an empty list so the configured
insufficient-history policy can handle it; the splitter default still raises.

The information constraints for inner origin `s` and outer origin `t` are:

- inner training: feature date < `s`, label available <= `s`;
- inner validation: feature date = `s` < `t`, label available <= `t`;
- outer validation: excluded from all candidate selection and preprocessing fits.

Feature availability declarations are checked again at `s`, including historical
training rows. Labels used for scoring can become available after `s`, provided they
are known by `t`. Missing or unreleased outer-training labels never enter the search.
As with forecasting, callers supply vintage-correct features.

`search.select_parameters` evaluates sklearn `ParameterGrid` candidates with fresh
pipelines per inner split. Fixed model parameters are merged with each candidate,
which takes precedence on shared keys. Loss is computed in original target units,
then averaged equally across validation dates. Search transforms must be stateless
and pointwise. The first candidate in deterministic grid order wins an exact tie.
Model seeds retain their registry/configuration semantics.

The forecaster refits the winning parameters on all outer training rows. It continues
to parallelize at the model level only. Candidate errors fail the run with context.
Insufficient inner history raises by default; explicit `use_fixed` records a selected
fallback with undefined score and no candidate fits. No partial search is performed.

`ForecastResult.search_results` stores one row per candidate, outer fold and model,
including parameters, split losses, score, selection, status and elapsed seconds.
`search_audit` stores inner split boundaries and label availability once per outer
fold, shared across models. Timings are observational and not deterministic.
Experiment schema 2 persists both tables and includes them in HTML reports; loading
schema 1 creates empty search tables. Grid search is disabled by default and does
not alter the fixed-parameter prediction path.
