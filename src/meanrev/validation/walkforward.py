"""Rolling walk-forward validation, replacing the original notebook's single static 75/25 split.

Also implements the two realizable execution conventions discussed for SPY (the notebook's
same-close approach isn't tradeable live without approximation):
  - convention "B": signal known after close(t-1), traded via a close(t-1)->close(t) return
    (requires a market-on-close order using a near-close price proxy - the notebook's approach).
  - convention "A": signal known after close(t-1), traded via an open(t)->open(t+1) return
    (fully realizable with a plain market-on-open order, at the cost of ~1 extra day of decay).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from meanrev.calendar import ANNUALIZATION_FACTOR


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(dates: pd.DatetimeIndex, train_sessions: int = 756, test_sessions: int = 126, step_sessions: int = 126) -> list[Fold]:
    dates = pd.DatetimeIndex(sorted(dates))
    n = len(dates)
    folds = []
    start = 0
    while start + train_sessions + test_sessions <= n:
        train = dates[start : start + train_sessions]
        test = dates[start + train_sessions : start + train_sessions + test_sessions]
        folds.append(Fold(train[0], train[-1], test[0], test[-1]))
        start += step_sessions
    return folds


def strategy_returns(df: pd.DataFrame, signal: pd.Series, convention: str = "A", cost_bps: float = 1.0) -> pd.Series:
    """Net daily log-returns dated at the signal date t, for the trade taken using signal
    known as of t's close, realized per `convention`. `cost_bps` is a per-side round-half-spread
    + slippage assumption (default 1bp, conservative for SPY's ~0.08bp quoted half-spread).

    signal_t is derived from data through t's close, so it can only be applied to returns that
    occur strictly after t: convention "B" captures close(t)->close(t+1) (needs a same-day MOC
    order using a near-close proxy for t's close); convention "A" captures open(t+1)->open(t+2)
    (a plain next-day MOO order, one extra session of signal decay but no intraday approximation).
    """
    if convention == "B":
        gross = signal * np.log(df["close"].shift(-1) / df["close"])
    elif convention == "A":
        gross = signal * np.log(df["open"].shift(-2) / df["open"].shift(-1))
    else:
        raise ValueError(f"unknown convention {convention!r}, expected 'A' or 'B'")

    # fillna(0) treats the pre-signal warm-up period as flat, so entering the first real
    # position is counted as turnover rather than propagating NaN through every later cost.
    turnover = signal.fillna(0.0).diff().abs()
    cost = turnover * (cost_bps / 10000)
    return (gross - cost).rename(f"net_return_{convention}")


def conditional_spread_bps(close: pd.Series) -> float:
    """E[return | yesterday down] - E[return | yesterday up], in bps: the core mean-reversion
    effect size from the original notebook's groupby analysis, over this `close` window."""
    ret = np.log(close / close.shift(1)).dropna()
    lag_dir = np.sign(ret.shift(1))
    down = ret[lag_dir < 0]
    up = ret[lag_dir > 0]
    return (down.mean() - up.mean()) * 10000


def fold_metrics(test_returns: pd.Series) -> dict:
    test_returns = test_returns.dropna()
    if test_returns.empty or test_returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = test_returns.mean() / test_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    return {
        "n_days": len(test_returns),
        "mean_daily_bps": test_returns.mean() * 10000,
        "sharpe": sharpe,
        "hit_rate": (test_returns > 0).mean(),
        "cum_return": np.exp(test_returns.sum()) - 1,
    }


def run_walk_forward(df: pd.DataFrame, signal: pd.Series, convention: str = "A", cost_bps: float = 1.0, **fold_kwargs) -> pd.DataFrame:
    """One row per fold: the conditional edge measured in-window (train_spread_bps) alongside
    what trading the out-of-window fold actually would have earned net of costs."""
    net_returns = strategy_returns(df, signal, convention=convention, cost_bps=cost_bps)
    folds = make_folds(df.index, **fold_kwargs)
    rows = []
    for f in folds:
        test_slice = net_returns.loc[f.test_start : f.test_end]
        train_close = df["close"].loc[f.train_start : f.train_end]
        row = fold_metrics(test_slice)
        row["train_spread_bps"] = conditional_spread_bps(train_close)
        row.update(train_start=f.train_start, train_end=f.train_end, test_start=f.test_start, test_end=f.test_end)
        rows.append(row)
    return pd.DataFrame(rows)
