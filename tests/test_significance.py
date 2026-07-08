import functools

import numpy as np
import pandas as pd

from meanrev.validation import significance as sig
from meanrev.validation import walkforward as wf


def test_conditional_ttest_matches_manual_computation():
    closes = [100, 102, 101, 108, 100, 105, 98]
    close = pd.Series(closes, index=pd.date_range("2024-01-01", periods=len(closes)))
    result = sig.conditional_ttest(close)

    ret = np.log(close / close.shift(1)).dropna()
    lag_dir = np.sign(ret.shift(1))
    down = ret[lag_dir < 0].dropna()
    up = ret[lag_dir > 0].dropna()
    assert result["n_down"] == len(down)
    assert result["n_up"] == len(up)
    assert np.isclose(result["mean_after_down_bps"], down.mean() * 10000)


def test_block_bootstrap_ci_contains_sample_mean():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0002, 0.01, 500))
    result = sig.block_bootstrap_ci(returns, block_size=10, n_boot=500, rng=rng)
    assert result["ci_low_bps"] < result["mean_daily_bps"] < result["ci_high_bps"]


def test_permutation_test_gives_high_pvalue_for_pure_noise():
    # random walk with no true relationship between signal and forward return -> most
    # permutations should do about as well as the "real" one (p-value not tiny).
    rng = np.random.default_rng(1)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    df = pd.DataFrame({"close": close, "open": close})
    signal = pd.Series(rng.choice([-1, 1], size=n), index=idx)

    fn = functools.partial(wf.strategy_returns, convention="B", cost_bps=0.0)
    result = sig.permutation_test(df, signal, fn, n_perm=200, rng=rng)
    assert result["p_value"] > 0.01
