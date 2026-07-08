"""Phase 1 Gate: is the lag-1 direction-fade signal (from the original BCH notebook) actually
tradeable on SPY, net of costs, out-of-sample? Runs the long/short and long-only variants
through walk-forward validation plus significance tests and prints a go/no-go verdict.
"""

from meanrev.data import historical
from meanrev.signal.mean_reversion import compute_signal
from meanrev.validation import gate

SYMBOL = "SPY"
COST_BPS = 1.0  # conservative all-in per-side assumption for SPY market orders


def main():
    df = historical.load(SYMBOL, start="1993-01-01")
    print(f"Loaded {len(df)} sessions of {SYMBOL} data, {df.index.min().date()} to {df.index.max().date()}")

    results = [
        gate.evaluate(df, compute_signal(df["close"]), f"{SYMBOL}: base rule (long/short fade)", cost_bps=COST_BPS),
        gate.evaluate(df, compute_signal(df["close"], long_only=True), f"{SYMBOL}: long-only variant (fade down days only)", cost_bps=COST_BPS),
    ]

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(f"  {r['label']}: {'GO' if r['passed'] else 'NO-GO'}")


if __name__ == "__main__":
    main()
