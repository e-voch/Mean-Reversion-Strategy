"""The lag-1 direction-fade signal, ported from the original BCH notebook.

Pure function, no I/O: given a close price series, returns a target position per date.
This is the single shared signal implementation used by both the backtest engine and
(eventually) the live runner, so backtest and live logic can never drift apart.
"""

import numpy as np
import pandas as pd


def compute_signal(close: pd.Series, long_only: bool = False) -> pd.Series:
    """Target position for the day *following* each return, in {-1, 0, +1} (or {0, +1} if long_only).

    Signal dated t is derived from the close-to-close return ending at t, i.e. it only uses
    data available through t's close - no lookahead.
    """
    log_return = np.log(close / close.shift(1))
    direction = np.sign(log_return)
    signal = -1 * direction
    if long_only:
        signal = signal.clip(lower=0)
    return signal.rename("signal")


def compute_signal_vol_conditioned(
    close: pd.Series,
    long_only: bool = False,
    vol_window: int = 20,
    vol_rank_window: int = 252,
    vol_rank_threshold: float = 0.5,
) -> pd.Series:
    """Base fade signal, but active only when short-term realized vol is elevated.

    Rationale: walk-forward folds show the fade rule earns most of its return in high-vol
    regimes (2008-09, 2020 H1, 2022 H2) - reversal is compensation for providing liquidity,
    and that compensation is only meaningful when markets are stressed.

    Point-in-time: vol at t uses returns through t, and its rank is taken against the trailing
    `vol_rank_window` distribution of that same vol series - no future data involved.
    """
    log_return = np.log(close / close.shift(1))
    vol = log_return.rolling(vol_window).std()
    vol_rank = vol.rolling(vol_rank_window).rank(pct=True)
    active = (vol_rank > vol_rank_threshold).astype(float)
    return (compute_signal(close, long_only=long_only) * active).rename("signal")
