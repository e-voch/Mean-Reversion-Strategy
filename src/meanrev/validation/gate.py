"""Gate #1 evaluation: walk-forward + significance, judged against fixed go/no-go thresholds.

Shared by every validation script so a signal variant can't be judged by a friendlier
yardstick than the one that failed its predecessor.
"""

import functools

import pandas as pd

from meanrev.calendar import ANNUALIZATION_FACTOR
from meanrev.validation import significance as sig
from meanrev.validation import walkforward as wf

THRESHOLDS = {
    "min_sharpe_recent": 0.5,
    "max_pvalue": 0.05,
    "min_pct_positive_folds": 0.60,
    "min_edge_vs_cost": 2.0,
}


def evaluate(
    df: pd.DataFrame,
    signal: pd.Series,
    label: str,
    cost_bps: float = 1.0,
    periods_per_year: int = ANNUALIZATION_FACTOR,
    recent_years: int = 10,
    n_perm: int = 10_000,
    convention: str = "A",
    verbose: bool = True,
    **fold_kwargs,
) -> dict:
    folds = wf.run_walk_forward(
        df, signal, convention=convention, cost_bps=cost_bps, periods_per_year=periods_per_year, **fold_kwargs
    )
    cutoff = df.index.max() - pd.DateOffset(years=recent_years)
    recent = folds[folds["test_start"] >= cutoff]

    sharpe_recent = recent["sharpe"].mean()
    pct_positive = (recent["cum_return"] > 0).mean()
    edge_vs_cost = recent["mean_daily_bps"].mean() / cost_bps

    # Significance is judged over the same recent window as the other gates - a full-history
    # p-value would let a strong-but-dead early era paper over a decayed recent one.
    fn = functools.partial(wf.strategy_returns, convention=convention, cost_bps=cost_bps)
    recent_start = recent["test_start"].min()
    perm = sig.permutation_test(df.loc[recent_start:], signal.loc[recent_start:], fn, n_perm=n_perm)

    passed = (
        sharpe_recent >= THRESHOLDS["min_sharpe_recent"]
        and pct_positive >= THRESHOLDS["min_pct_positive_folds"]
        and perm["p_value"] < THRESHOLDS["max_pvalue"]
        and edge_vs_cost >= THRESHOLDS["min_edge_vs_cost"]
    )

    result = {
        "label": label,
        "folds": folds,
        "sharpe_recent": sharpe_recent,
        "pct_positive": pct_positive,
        "p_value": perm["p_value"],
        "edge_vs_cost": edge_vs_cost,
        "passed": passed,
    }
    if verbose:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        print(folds[["test_start", "test_end", "train_spread_bps", "sharpe", "mean_daily_bps", "hit_rate", "cum_return"]].to_string(index=False))
        print(f"\nRecent ({recent_years}y) OOS mean fold Sharpe: {sharpe_recent:.3f}  (gate: >= {THRESHOLDS['min_sharpe_recent']})")
        print(f"Recent OOS pct positive folds:      {pct_positive:.0%}  (gate: >= {THRESHOLDS['min_pct_positive_folds']:.0%})")
        print(f"Recent permutation p-value:         {perm['p_value']:.4f}  (gate: < {THRESHOLDS['max_pvalue']}, n_perm={n_perm})")
        print(f"Net edge vs. assumed cost:          {edge_vs_cost:.2f}x  (gate: >= {THRESHOLDS['min_edge_vs_cost']}x, cost={cost_bps}bps/side)")
        print(f"\nVERDICT: {'GO' if passed else 'NO-GO'}")
    return result
