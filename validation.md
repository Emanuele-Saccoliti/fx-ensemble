# Validation evidence

Validation date: 2026-09-05

Runtime used:

- Python 3.12.0
- NumPy 2.2.6
- pandas 2.3.0
- scikit-learn 1.7.2
- SciPy 1.16.0

## Automated tests

Command:

```bash
python -m pytest -q
```

Result: **13 passed** in 3.45 seconds.

The suite covers expanding-window chronology, delayed label availability, adaptive
ensemble information sets, deterministic parallel forecasts, positive log-target
volatility forecasts, feature release-date rejection, custom model registration,
forecast metrics, covariance shrinkage, constrained allocation, transaction costs,
and post-return holdings drift.

## Distribution smoke test

The wheel was built without downloading dependencies, installed into an isolated
target directory, and imported from that installed location. The reported version
was `0.1.0`. The installed wheel then ran
`examples/volatility_forecast.py` end to end and produced 72 OOS observations for
each of three base models and two ensemble rules.

Artifact:

```text
dist/fxensemble-0.1.0-py3-none-any.whl
SHA-256 680a8671e93b93c2cc9a78df59cfc3818becf1527d686a6f76c9c25d4d440917
```

The wheel contains all nine package modules plus the `py.typed` marker. A source
scan found no absolute user paths and no dependency on the `TEST2` research folder.
