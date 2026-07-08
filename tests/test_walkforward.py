import numpy as np
import pandas as pd

from meanrev.signal.mean_reversion import compute_signal
from meanrev.validation import walkforward as wf


def _df(closes, opens=None):
    idx = pd.date_range("2024-01-01", periods=len(closes))
    opens = closes if opens is None else opens
    return pd.DataFrame({"close": closes, "open": opens}, index=idx)


def test_convention_b_uses_return_strictly_after_signal_date():
    closes = [100, 102, 101, 108, 100]
    df = _df(closes)
    signal = compute_signal(df["close"])
    net = wf.strategy_returns(df, signal, convention="B", cost_bps=0.0)

    # signal at day1 (=-1, fading the 100->102 up move) must apply to day1->day2 (102->101), not day0->day1.
    expected = -1 * np.log(closes[2] / closes[1])
    assert np.isclose(net.iloc[1], expected)


def test_convention_a_uses_open_strictly_after_signal_date():
    closes = [100, 102, 101, 108, 100]
    opens = [99, 101, 103, 107, 99]
    df = _df(closes, opens)
    signal = compute_signal(df["close"])
    net = wf.strategy_returns(df, signal, convention="A", cost_bps=0.0)

    # signal at day1 is known only after day1's close -> traded open(day2)->open(day3).
    expected = signal.iloc[1] * np.log(opens[3] / opens[2])
    assert np.isclose(net.iloc[1], expected)


def test_no_lookahead_regression_guard():
    """Regression guard for a bug found during development: applying signal_t to the same
    return that generated it makes gross return deterministically -|return| every day. That
    bug produced ~-50bps/day "returns" that were actually just -sign(r)*r in disguise."""
    closes = [100, 105, 95, 110, 90, 120]
    df = _df(closes)
    signal = compute_signal(df["close"])
    net = wf.strategy_returns(df, signal, convention="B", cost_bps=0.0).dropna()

    generating_return = np.log(df["close"] / df["close"].shift(1))
    forbidden = (-signal.abs() * generating_return.abs()).loc[net.index]
    assert not np.allclose(net.to_numpy(), forbidden.to_numpy())


def test_make_folds_produces_contiguous_non_overlapping_test_windows():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    folds = wf.make_folds(dates, train_sessions=5, test_sessions=5, step_sessions=5)
    assert len(folds) == 3  # starts at 0, 5, 10 all satisfy start + train + test <= 20
    assert folds[0].test_end < folds[1].test_start


def test_conditional_spread_matches_manual_groupby():
    closes = [100, 102, 101, 108, 100, 105]
    close = pd.Series(closes, index=pd.date_range("2024-01-01", periods=len(closes)))
    spread = wf.conditional_spread_bps(close)

    ret = np.log(close / close.shift(1)).dropna()
    lag_dir = np.sign(ret.shift(1))
    expected = (ret[lag_dir < 0].mean() - ret[lag_dir > 0].mean()) * 10000
    assert np.isclose(spread, expected)


def test_resample_weekly_uses_first_open_last_close_and_real_session_dates():
    idx = pd.bdate_range("2024-01-01", "2024-01-19")  # 3 full weeks
    df = pd.DataFrame({"open": np.arange(len(idx), dtype=float) + 1, "close": np.arange(len(idx), dtype=float) + 100}, index=idx)
    weekly = wf.resample_weekly(df)

    assert len(weekly) == 3
    assert weekly.index.isin(df.index).all()
    assert weekly.iloc[0]["open"] == df.iloc[0]["open"]        # Monday's open
    assert weekly.iloc[0]["close"] == df.iloc[4]["close"]      # Friday's close
