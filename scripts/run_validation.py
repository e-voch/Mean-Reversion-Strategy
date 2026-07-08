"""Phase 1 Gate: is the lag-1 direction-fade signal (from the original BCH notebook) actually
tradeable on SPY, net of costs, out-of-sample? Runs both execution conventions and both the
long/short and long-only variants through walk-forward validation plus significance tests,
and prints a go/no-go verdict against the thresholds in the plan.
"""

import functools

import pandas as pd

from meanrev.data import historical
from meanrev.signal.mean_reversion import compute_signal
from meanrev.validation import significance as sig
from meanrev.validation import walkforward as wf

SYMBOL = "SPY"
COST_BPS = 1.0  # conservative all-in per-side assumption for SPY market orders
N_PERM = 10_000

GATE = {
    "min_sharpe_last_10y": 0.5,
    "max_pvalue": 0.05,
    "min_pct_positive_folds": 0.60,
}


def last_n_years(folds, years, end_date):
    cutoff = end_date - pd.DateOffset(years=years)
    return folds[folds["test_start"] >= cutoff]


def evaluate(df, signal, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    folds = wf.run_walk_forward(df, signal, convention="A", cost_bps=COST_BPS)
    recent = last_n_years(folds, 10, df.index.max())

    sharpe_recent = recent["sharpe"].mean()
    pct_positive = (recent["cum_return"] > 0).mean()
    edge_vs_cost = recent["mean_daily_bps"].mean() / COST_BPS

    # Significance must be judged over the same recent window as the Sharpe/hit-rate gates above -
    # a full-history permutation test would let a strong 1990s/2000s effect paper over a dead recent one.
    fn = functools.partial(wf.strategy_returns, convention="A", cost_bps=COST_BPS)
    recent_start = recent["test_start"].min()
    perm_recent = sig.permutation_test(df.loc[recent_start:], signal.loc[recent_start:], fn, n_perm=N_PERM)
    perm_full = sig.permutation_test(df, signal, fn, n_perm=N_PERM)

    print(folds[["test_start", "test_end", "train_spread_bps", "sharpe", "mean_daily_bps", "hit_rate", "cum_return"]].to_string(index=False))
    print(f"\nLast-10y OOS mean fold Sharpe:   {sharpe_recent:.3f}  (gate: >= {GATE['min_sharpe_last_10y']})")
    print(f"Last-10y OOS pct positive folds: {pct_positive:.0%}  (gate: >= {GATE['min_pct_positive_folds']:.0%})")
    print(f"Last-10y permutation p-value:    {perm_recent['p_value']:.4f}  (gate: < {GATE['max_pvalue']}, n_perm={N_PERM})")
    print(f"Net edge vs. assumed cost:       {edge_vs_cost:.2f}x  (gate: >= 2x)")
    print(f"[context only, not gated] full-history ({df.index.min().date()}-present) permutation p-value: {perm_full['p_value']:.4f}")

    passed = (
        sharpe_recent >= GATE["min_sharpe_last_10y"]
        and pct_positive >= GATE["min_pct_positive_folds"]
        and perm_recent["p_value"] < GATE["max_pvalue"]
        and edge_vs_cost >= 2.0
    )
    print(f"\nVERDICT: {'GO' if passed else 'NO-GO'}")
    return passed


def main():
    df = historical.load(SYMBOL, start="1993-01-01")
    print(f"Loaded {len(df)} sessions of {SYMBOL} data, {df.index.min().date()} to {df.index.max().date()}")

    long_short = compute_signal(df["close"])
    long_only = compute_signal(df["close"], long_only=True)

    results = {
        "long/short": evaluate(df, long_short, f"{SYMBOL}: base rule (long/short fade)"),
        "long-only": evaluate(df, long_only, f"{SYMBOL}: long-only variant (fade down days only)"),
    }

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, passed in results.items():
        print(f"  {name}: {'GO' if passed else 'NO-GO'}")


if __name__ == "__main__":
    main()
