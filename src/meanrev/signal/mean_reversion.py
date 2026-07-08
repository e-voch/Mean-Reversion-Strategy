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
    data available through t's close — no lookahead.
    """
    log_return = np.log(close / close.shift(1))
    direction = np.sign(log_return)
    signal = -1 * direction
    if long_only:
        signal = signal.clip(lower=0)
    return signal.rename("signal")
