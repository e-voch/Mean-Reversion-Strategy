"""Statistical significance tests for the mean-reversion effect and the resulting strategy.

These exist because a pattern found via one groupby on one holdout split (the original
notebook's approach) is not strong evidence for a real, tradeable effect.
"""

import numpy as np
import pandas as pd
from scipy import stats

from meanrev.calendar import ANNUALIZATION_FACTOR


def conditional_ttest(close: pd.Series) -> dict:
    """Welch's t-test: next-day return after a down day vs after an up day."""
    ret = np.log(close / close.shift(1)).dropna()
    lag_dir = np.sign(ret.shift(1))
    down = ret[lag_dir < 0].dropna()
    up = ret[lag_dir > 0].dropna()
    t_stat, p_value = stats.ttest_ind(down, up, equal_var=False)
    return {
        "n_down": len(down),
        "n_up": len(up),
        "mean_after_down_bps": down.mean() * 10000,
        "mean_after_up_bps": up.mean() * 10000,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def block_bootstrap_ci(
    returns: pd.Series,
    block_size: int = 20,
    n_boot: int = 5000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Circular block bootstrap CI on the mean daily return, preserving autocorrelation
    (a plain iid bootstrap would understate the true uncertainty for return series like this)."""
    values = returns.dropna().to_numpy()
    n = len(values)
    rng = rng or np.random.default_rng()
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i, :, None] + np.arange(block_size)[None, :]) % n
        means[i] = values[idx.ravel()][:n].mean()
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {
        "mean_daily_bps": values.mean() * 10000,
        "ci_low_bps": lo * 10000,
        "ci_high_bps": hi * 10000,
        "n_boot": n_boot,
    }


def permutation_test(
    df: pd.DataFrame,
    signal: pd.Series,
    strategy_returns_fn,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Circularly shift the signal relative to price/return data n_perm times and recompute the
    strategy under each shift; p-value = fraction of shifts whose Sharpe >= the realized Sharpe.
    Directly answers "is this specific signal/return alignment special, or would any similarly
    autocorrelated series do?" `strategy_returns_fn(df, signal) -> pd.Series` should already be
    bound to a convention/cost_bps (e.g. via functools.partial).
    """
    rng = rng or np.random.default_rng()

    real = strategy_returns_fn(df, signal).dropna()
    real_sharpe = real.mean() / real.std() * np.sqrt(ANNUALIZATION_FACTOR)

    values = signal.to_numpy()
    n = len(values)
    shifts = rng.integers(1, n, size=n_perm)
    null_sharpes = np.empty(n_perm)
    for i, k in enumerate(shifts):
        shifted = pd.Series(np.roll(values, k), index=signal.index)
        r = strategy_returns_fn(df, shifted).dropna()
        null_sharpes[i] = r.mean() / r.std() * np.sqrt(ANNUALIZATION_FACTOR) if r.std() > 0 else 0.0

    p_value = (null_sharpes >= real_sharpe).mean()
    return {
        "real_sharpe": real_sharpe,
        "null_mean_sharpe": null_sharpes.mean(),
        "p_value": p_value,
        "n_perm": n_perm,
    }
