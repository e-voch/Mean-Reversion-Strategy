# Mean-Reversion Strategy - Project Documentation

*Last updated: 2026-07-08*

---

## 1. What this project set out to achieve

### 1.1 Origin

The project began as a single research notebook (`notebooks/01_bch_original.ipynb`) that found a
**lag-1 direction-fade mean-reversion pattern** in Bitcoin Cash (BCH) daily price data:

> If yesterday's price move was down, today's move tends to be up - and vice versa.
> So: fade yesterday's direction. `signal = -1 × sign(yesterday's return)`

The notebook validated this with a single 75/25 chronological train/test split, computed a win
rate, compound return, and Sharpe ratio, and layered a simple crypto exchange fee model
(maker/taker basis points) on top.

### 1.2 The goal

Turn that research finding into a **production-ready trading system for the S&P 500 (via the
SPY ETF)** - something that could be deployed to paper-trade and eventually live-trade with
real risk controls, rather than remaining a one-off notebook analysis.

### 1.3 The plan (phased, gated)

A full rollout plan was drafted covering repo architecture, data layer, signal validation,
backtest engine, risk management, execution via Alpaca, ops/monitoring, and testing. Its key
design feature was **go/no-go gates between phases**, with Phase 1 deliberately front-loading
the question that could kill the project:

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Repo scaffold, package structure | Tests green |
| 1 | **SPY signal validation** | Net Sharpe ≥ 0.5 (recent 10y OOS), permutation p < 0.05, ≥60% positive folds, edge ≥ 2× costs - **else stop** |
| 2 | Backtest engine | Engine reproduces notebook results; suite green |
| 3 | Risk framework (vol targeting, kill-switch) | Verified on 2008/2020 history |
| 4 | Paper trading on Alpaca | ≥60 sessions, clean reconciliation |
| 5 | Live with small capital | 3 months tracking within tolerance |

The rationale: short-term daily reversal in US large-caps is a *documented but decayed* anomaly
(strong pre-2013, weak since). Finding out whether it survives should cost a week of research,
not a full production build or real capital.

**The project stopped at the Phase 1 gate.** See Results (§4).

---

## 2. Project structure

```
Mean-Reversion-Strategy/
├── README.md                     # Quick start + summary of validation results
├── pyproject.toml                # Package metadata & dependencies (pip install -e ".[dev]")
├── .env.example                  # Template for machine-specific env config
├── .gitignore                    # Excludes data/, logs/, .local/, .env
├── docs/
│   └── PROJECT_DOCUMENTATION.md  # This file
├── notebooks/
│   └── 01_bch_original.ipynb     # The original BCH research notebook, preserved as-is
├── scripts/
│   ├── run_validation.py         # Phase 1 gate: base rule on SPY (long/short + long-only)
│   └── run_pivots.py             # Pivot variants: vol-regime, weekly, crypto re-validation
├── src/meanrev/                  # The installable package - all logic lives here, not in notebooks
│   ├── __init__.py               # TLS workaround auto-setup (see §6.2)
│   ├── calendar.py               # NYSE trading calendar helpers
│   ├── data/
│   │   └── historical.py         # yfinance loader: adjusted+raw prices, parquet cache, QC gate
│   ├── signal/
│   │   └── mean_reversion.py     # The signal functions (pure, no I/O)
│   └── validation/
│       ├── walkforward.py        # Rolling folds, execution conventions, cost model
│       ├── significance.py       # t-test, block bootstrap, permutation test
│       └── gate.py               # Shared go/no-go evaluation with fixed thresholds
├── tests/                        # 29 unit tests (pytest)
└── data/                         # (gitignored) parquet price cache
```

### Design principles

1. **One signal implementation.** `signal/mean_reversion.py` is a pure, I/O-free function meant
   to be shared by research, backtest, and (had it gone live) the production runner - so
   backtest and live logic can never drift apart.
2. **Research imports production, never the reverse.** Notebooks and scripts call package
   functions; no logic lives only in a notebook.
3. **One gate, fixed thresholds.** Every strategy variant is judged by
   `validation/gate.py` with identical criteria - a new variant cannot be judged by a
   friendlier yardstick than the one that failed its predecessor.
4. **All date arithmetic goes through `calendar.py`.** Equities trade ~252 sessions/year with
   holidays and early closes; naive `timedelta(days=1)` math silently corrupts alignment.

---

## 3. Important files in detail

### `src/meanrev/signal/mean_reversion.py`
- `compute_signal(close, long_only=False)` - the base rule. Signal dated *t* uses only data
  through *t*'s close (no lookahead). Returns target position ∈ {-1, 0, +1}.
- `compute_signal_vol_conditioned(...)` - pivot variant: signal active only when 20-day
  realized vol ranks above the median of its trailing 1-year distribution (point-in-time
  rolling rank, no future data).

### `src/meanrev/validation/walkforward.py`
- `make_folds(...)` - rolling ~3y-train / 6m-test / 6m-step windows, replacing the notebook's
  single 75/25 split (which can be dominated by one lucky/dead era).
- `strategy_returns(df, signal, convention, cost_bps)` - computes net returns under a
  **realizable execution convention**. The notebook implicitly traded at the same close that
  generated the signal, which is impossible live. Two honest alternatives are modeled:
  - **Convention A**: signal after close(t) → trade market-on-open(t+1) → P&L open(t+1)→open(t+2)
  - **Convention B**: near-close proxy signal → market-on-close(t) → P&L close(t)→close(t+1)
- `resample_weekly(df)` - weekly bars for the weekly-frequency pivot, dated by the week's last
  session so signals remain point-in-time.
- Costs are modeled as per-side bps on turnover (1bp/side SPY, 5bps/side crypto - both
  conservative).

### `src/meanrev/validation/significance.py`
- `conditional_ttest` - Welch's t-test: is E[return | yesterday down] really different from
  E[return | yesterday up]?
- `block_bootstrap_ci` - circular block bootstrap CI on mean daily return (preserves
  autocorrelation an iid bootstrap would destroy).
- `permutation_test` - circularly shifts the signal against returns 10,000× and compares the
  realized Sharpe to the null distribution. This is the test that matters most: it answers
  *"is this particular signal/return alignment special, or would any similarly-shaped series
  do as well?"* - and it is what exposed the long-only variants (§4.3).

### `src/meanrev/validation/gate.py`
The Gate #1 thresholds, applied identically to every variant:
- Recent (10y equities / 5y crypto) out-of-sample mean fold Sharpe ≥ 0.5
- ≥ 60% of recent folds net-positive
- Permutation p-value < 0.05 **on the same recent window** (a full-history p-value would let a
  strong-but-dead early era paper over a decayed recent one)
- Net edge ≥ 2× assumed round-trip cost

### `src/meanrev/data/historical.py`
- Downloads **both adjusted and raw close** via yfinance. Adjusted close is mandatory for
  return/signal math - SPY pays dividends, and raw close shows artificial drops on ex-dividend
  dates that would corrupt log returns and generate false "down day" signals.
- Parquet cache with range checking (re-downloads if the cached range doesn't cover the request).
- A QC gate that **raises** (never silently continues) on NaNs, non-positive prices, missing
  NYSE sessions, or rows on non-session dates. `nyse_calendar=False` relaxes the session check
  for 24/7 assets (crypto).

### `tests/` (29 tests)
Unit tests for signal correctness (including NaN/first-day edges), calendar behavior (holidays,
early closes, `sqrt(252)` not `sqrt(365)`), fold construction, cost math, QC rejection paths,
cache behavior, and - most importantly - **no-lookahead regression guards**, written after a
real lookahead bug was caught during development (§5).

---

## 4. Results

### 4.1 Phase 1: base rule on SPY - NO-GO

Full SPY history (1993–2026), walk-forward, net of 1bp/side costs, Convention A:

| Variant | Recent-10y OOS mean fold Sharpe | % positive folds | Permutation p | Edge vs cost | Verdict |
|---|---|---|---|---|---|
| Long/short fade | **-0.36** | 42% | 0.45 | -1.0× | **NO-GO** |
| Long-only fade | 0.57 | 68% | 0.45 | 2.2× | **NO-GO** |

The fold-by-fold history shows the textbook decay pattern: the conditional return spread
(`train_spread_bps`) was large and reliable from the late 1990s through ~2012 (peaking at
26–31bps/day around 2008–09), then shrank and destabilized. The strategy's recent OOS
performance is indistinguishable from - or worse than - noise.

### 4.2 Pivots - all NO-GO

Three documented pivots were run through the identical gate:

| Variant | Recent OOS Sharpe | Permutation p | Verdict |
|---|---|---|---|
| SPY daily, high-vol regime only (long/short) | -0.02 | 0.21 | NO-GO |
| SPY daily, high-vol regime only (long-only) | 0.82 | 0.45 | NO-GO |
| SPY weekly fade (long/short) | 0.15 | 0.06 | NO-GO |
| SPY weekly fade (long-only) | 1.19 | 0.23 | NO-GO |
| BTC-USD daily fade (5y window, 5bps costs) | -0.66 | 0.77 | NO-GO |
| ETH-USD daily fade (5y window, 5bps costs) | -0.03 | 0.28 | NO-GO |

### 4.3 The key insight: beta masquerading as alpha

The long-only variants *look* attractive (weekly long-only: Sharpe 1.19, 84% positive folds).
This is precisely the trap the permutation test exists to catch: a signal that is long roughly
half the time earns roughly half the equity risk premium **regardless of when it is long**.
Circularly shifting the signal - destroying the timing while preserving the fraction of time
in the market - produces nearly equivalent results (p = 0.23–0.45). The return is **market
beta, not reversal timing**. The same exposure is available more cheaply by holding SPY at
half size with zero trading.

### 4.4 The original finding, revisited

The BCH pattern in the original notebook was real for its era. But the same rig applied to
BTC/ETH over the recent 5 years shows the crypto daily-reversal effect has decayed too - BTC
has actually flipped toward momentum (21% positive folds for the fade rule). The notebook
measured a genuine but perishable inefficiency, near the end of its shelf life.

### 4.5 Conclusion

**The lag-1 direction-fade family has no tradeable edge remaining in any market or frequency
tested.** Per the plan's own stop rule, no backtest engine, risk framework, or execution
infrastructure was built on top of it. This outcome was explicitly anticipated in the plan and
is the gate system working as designed: the discovery cost ~a day of work, not a production
build or real losses.

---

## 5. Bugs caught during development (why the testing rigor mattered)

Three bugs were caught before any result was trusted - each would have silently corrupted
conclusions:

1. **Lookahead bug**: the first implementation paired signal *t* with the same return that
   generated it, making gross return deterministically `-|r|` every day (~-50bps/day of fake
   losses). Caught by hand-checking fold output; now covered by a dedicated no-lookahead
   regression test. The inverse bug (positive lookahead) would have produced a fake GO.
2. **NaN propagation**: the turnover/cost series propagated the signal's warm-up NaN through
   every subsequent cost value.
3. **Stale cache**: the data loader served a previously-cached narrow date range regardless of
   the range requested - the first "full-history" validation actually ran on 2015–2024 data.

Lesson encoded in the repo: **in backtest code, bugs are asymmetric** - they rarely announce
themselves, they just make results look slightly better or worse than reality. Every result
should be gated behind tests for the failure mode that would most flatter it.

---

## 6. Setup & runbook

### 6.1 Install

```
pip install -e ".[dev]"
pytest                          # 29 tests, all green expected
python scripts/run_validation.py   # Phase 1 gate report (SPY base rule)
python scripts/run_pivots.py       # Pivot variants report
```

Python ≥3.11. Dependencies: pandas, numpy, yfinance, scipy, pandas-market-calendars, pyarrow,
truststore (pydantic/dotenv/structlog are declared for the later phases that weren't built).

### 6.2 Windows + antivirus TLS interception

On machines where an antivirus product (Norton, BullGuard) performs HTTPS scanning, Python's
strict TLS validation rejects the AV's proxy certificate (`Basic Constraints of CA cert not
marked critical`), breaking yfinance. The fix used here:

1. Export the AV's root cert(s) from the Windows trust store (`Cert:\LocalMachine\Root`) as PEM.
2. Concatenate them onto certifi's bundle and save as `.local/ca-bundle.pem` (gitignored).
3. `meanrev/__init__.py` auto-detects that file and sets `CURL_CA_BUNDLE` /
   `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`; it also calls `truststore.inject_into_ssl()` so
   stdlib/requests use the OS trust store directly.

### 6.3 Data cache

Price data is cached under `data/` as parquet (gitignored). Delete a file or pass
`refresh=True` to `historical.load()` to force a re-download. The loader re-downloads
automatically when the cached range doesn't cover the requested range.

---

## 7. Future work

### 7.1 If continuing signal research (recommended framing: pipeline, not strategy)

The durable asset of this project is the **validation rig**, not any signal. Testing a new
daily-bar signal idea is now a ~20-line script: load data → build a signal series → call
`gate.evaluate()`. Candidate research directions, roughly ordered by expected shelf life:

- **Structurally-anchored mean reversion**: ETF share-class/duplicate pairs, ADR/local
  cross-listings, futures calendar spreads, ETF NAV dislocations. The reversion anchor is
  contractual rather than behavioral, so competition compresses margins rather than
  eliminating the relationship. Highest effort, longest shelf life.
- **Other signal classes through the same gate**: longer-horizon trend/momentum,
  seasonality effects, volatility-risk-premium harvesting.
- **Less efficient markets**: small caps, non-US markets, or crypto microstructure at
  intraday frequencies (requires new data infrastructure and much stricter cost modeling).

### 7.2 If a future signal passes the gate

The original phased plan resumes at Phase 2 and remains valid as written:
backtest engine with point-in-time discipline → vol-targeted sizing + drawdown kill-switch →
Alpaca paper trading (≥60 sessions) → risk-gated live capital. Two production notes from the
plan worth preserving:

- **Execution convention must match between backtest and live** (Convention A/B machinery in
  `walkforward.py` already models this).
- **A live strategy needs an edge-decay monitor**, not just entry validation: track the rolling
  conditional spread in production and de-allocate when it drifts below the cost floor -
  kill-switch logic for decay, not just drawdown.

### 7.3 Rig improvements (small, optional)

- Multiple-hypothesis correction: six variants were tested; a future sweep over many signals
  should control family-wise error (e.g., White's Reality Check / SPA test) or the gate's
  p<0.05 loses meaning.
- Deflated Sharpe ratio as an additional gate metric.
- Convention B early-close handling (1pm closes) if MOC execution is ever pursued.
- CI (GitHub Actions) to run the test suite on push.

---

## 8. Timeline summary

| Date | Event |
|---|---|
| (earlier) | Original BCH notebook research; pattern found; naive validation passed |
| 2026-07-08 | Full plan drafted; Phase 0 scaffold built; Phase 1 rig implemented with 29 tests |
| 2026-07-08 | Phase 1 gate: SPY base rule **NO-GO** (both variants) |
| 2026-07-08 | Pivots (vol-regime, weekly, BTC/ETH) - **all NO-GO**; project stopped at gate per plan |
