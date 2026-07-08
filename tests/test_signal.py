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
