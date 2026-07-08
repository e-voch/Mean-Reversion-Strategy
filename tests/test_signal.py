import numpy as np
import pandas as pd

from meanrev.signal.mean_reversion import compute_signal


def _close(prices):
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=len(prices)))


def test_first_day_is_nan():
    signal = compute_signal(_close([100, 102, 101]))
    assert np.isnan(signal.iloc[0])


def test_fades_up_and_down_moves():
    # 100->102 (up) -> fade to -1; 102->101 (down) -> fade to +1; 101->105 (up) -> fade to -1
    signal = compute_signal(_close([100, 102, 101, 105]))
    assert signal.iloc[1] == -1
    assert signal.iloc[2] == 1
    assert signal.iloc[3] == -1


def test_long_only_clips_short_signals_to_zero():
    signal = compute_signal(_close([100, 102, 101, 105]), long_only=True)
    assert signal.iloc[1] == 0
    assert signal.iloc[2] == 1
    assert signal.iloc[3] == 0


def test_zero_return_gives_zero_signal():
    signal = compute_signal(_close([100, 100, 105]))
    assert signal.iloc[1] == 0


def test_vol_conditioned_signal_is_zero_when_vol_ranks_low():
    from meanrev.signal.mean_reversion import compute_signal_vol_conditioned

    # 120 high-vol sessions then 40 calm ones, with the rank window still dominated by the
    # high-vol era at the end - so recent calm days must rank low and be gated to 0. (The rank
    # is regime-relative: once the high-vol era ages out of the window, calm days can rank
    # "high" among themselves again, which is intended behavior, not a bug.)
    prices, level = [], 100.0
    for i in range(120):
        level *= 1.05 if i % 2 == 0 else 0.95
        prices.append(level)
    for i in range(40):
        level *= 1.0001 if i % 2 == 0 else 0.9999
        prices.append(level)
    close = pd.Series(prices, index=pd.date_range("2020-01-01", periods=len(prices)))

    signal = compute_signal_vol_conditioned(close, vol_window=10, vol_rank_window=100)
    assert (signal.iloc[-20:] == 0).all()


def test_vol_conditioned_signal_matches_base_when_active():
    from meanrev.signal.mean_reversion import compute_signal, compute_signal_vol_conditioned

    rng = np.random.default_rng(0)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
        index=pd.date_range("2020-01-01", periods=400),
    )
    base = compute_signal(close)
    conditioned = compute_signal_vol_conditioned(close)
    active = (conditioned != 0) & conditioned.notna()
    assert active.any()
    assert (conditioned[active] == base[active]).all()
