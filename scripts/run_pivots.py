"""Pivot tests after the base rule's NO-GO on SPY, each through the same Gate #1 thresholds:

1. Vol-regime conditioning: fade rule active only when 20d realized vol is in the upper half
   of its trailing 1y distribution (the folds where the base rule made money were high-vol).
2. Weekly frequency: fade the prior *week's* direction (weekly reversal decayed more slowly
   than daily in the equity literature).
3. Crypto re-validation: the market where the effect was originally found (BCH notebook) -
   BTC/ETH daily, 365d annualization, crypto-realistic costs.
"""

from meanrev.data import historical
from meanrev.signal.mean_reversion import compute_signal, compute_signal_vol_conditioned
from meanrev.validation import gate
from meanrev.validation.walkforward import resample_weekly

EQUITY_COST_BPS = 1.0
CRYPTO_COST_BPS = 5.0  # taker fee + slippage on a major venue, conservative retail assumption

# ~3y train / 6m test expressed in weeks for the weekly variant
WEEKLY_FOLDS = dict(train_sessions=156, test_sessions=26, step_sessions=26)


def main():
    results = []

    spy = historical.load("SPY", start="1993-01-01")

    for long_only in (False, True):
        variant = "long-only" if long_only else "long/short"
        signal = compute_signal_vol_conditioned(spy["close"], long_only=long_only)
        results.append(
            gate.evaluate(spy, signal, f"SPY daily fade, high-vol regime only ({variant})", cost_bps=EQUITY_COST_BPS)
        )

    weekly = resample_weekly(spy)
    for long_only in (False, True):
        variant = "long-only" if long_only else "long/short"
        signal = compute_signal(weekly["close"], long_only=long_only)
        results.append(
            gate.evaluate(
                weekly, signal, f"SPY weekly fade ({variant})",
                cost_bps=EQUITY_COST_BPS, periods_per_year=52, **WEEKLY_FOLDS,
            )
        )

    for symbol in ("BTC-USD", "ETH-USD"):
        df = historical.load(symbol, start="2015-01-01", nyse_calendar=False)
        print(f"\nLoaded {len(df)} days of {symbol}, {df.index.min().date()} to {df.index.max().date()}")
        signal = compute_signal(df["close"])
        results.append(
            gate.evaluate(
                df, signal, f"{symbol} daily fade (long/short)",
                cost_bps=CRYPTO_COST_BPS, periods_per_year=365, recent_years=5,
            )
        )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(f"  {'GO   ' if r['passed'] else 'NO-GO'}  {r['label']}")


if __name__ == "__main__":
    main()
