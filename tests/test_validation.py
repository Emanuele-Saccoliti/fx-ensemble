import pandas as pd

from fxensemble import ExpandingWindowConfig, ExpandingWindowSplitter


def test_splitter_uses_only_realized_labels_and_steps_by_forecast_period():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    frame = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B"] * len(dates),
            "target": range(2 * len(dates)),
            "label_date": (dates + pd.offsets.MonthBegin(1)).repeat(2),
        }
    )
    splitter = ExpandingWindowSplitter(
        ExpandingWindowConfig(min_train_periods=3, step_periods=2)
    )

    folds = splitter.split(
        frame,
        date_column="date",
        target_column="target",
        label_available_column="label_date",
    )

    assert [fold.forecast_date for fold in folds] == [dates[3], dates[5], dates[7]]
    for fold in folds:
        train = frame.iloc[fold.train_indices]
        test = frame.iloc[fold.test_indices]
        assert (train["date"] < fold.forecast_date).all()
        assert (train["label_date"] <= fold.forecast_date).all()
        assert (test["date"] == fold.forecast_date).all()


def test_unavailable_historical_label_is_excluded():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "target": [1.0, 2.0, 3.0],
            "label_date": pd.to_datetime(["2020-02-01", "2021-01-01", "2020-04-01"]),
        }
    )
    folds = ExpandingWindowSplitter(
        ExpandingWindowConfig(min_train_periods=1, start="2020-03-01")
    ).split(
        frame,
        date_column="date",
        target_column="target",
        label_available_column="label_date",
    )

    train = frame.iloc[folds[0].train_indices]
    assert train["target"].tolist() == [1.0]
