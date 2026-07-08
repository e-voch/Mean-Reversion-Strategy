# Mean-Reversion-Strategy

A lag-1 direction-fade mean-reversion signal, originally discovered on Bitcoin Cash
(`notebooks/01_bch_original.ipynb`), validated for SPY under a walk-forward /
statistical-significance framework before any production trading code is built.

**Full project documentation — goals, structure, methodology, results, and future work — is in
[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).**

Status: **Phase 1 validation complete — NO-GO on the base rule and all pivots**, see below.

## Setup

```
pip install -e ".[dev]"
```

### Windows + antivirus HTTPS scanning (Norton/BullGuard)

Some antivirus products scan HTTPS traffic with a proxy certificate that fails strict TLS
verification (`Basic Constraints of CA cert not marked critical`), breaking `yfinance`. If you
hit this:

1. Export your AV's root CA cert(s) from `Cert:\LocalMachine\Root` (PowerShell `Get-ChildItem`)
   as PEM.
2. Append them to a copy of `certifi`'s bundle, saved as `.local/ca-bundle.pem` (gitignored).
3. `meanrev/__init__.py` auto-detects that file and points `CURL_CA_BUNDLE` /
   `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` at it — no other setup needed.

## Running the Phase 1 validation report

```
python scripts/run_validation.py
```

Downloads full-history SPY data, runs rolling walk-forward validation and a permutation
significance test for both the long/short and long-only signal variants, and prints a
go/no-go verdict against the thresholds in the plan.

**Result as of 2026-07-08**: both variants are **NO-GO**. The long/short base rule nets
negative over the last 10 years of out-of-sample folds (mean Sharpe -0.36, edge below assumed
costs). The long-only variant (fade down-days only) has a more plausible point-estimate Sharpe
(0.57) but fails the significance gate (permutation p=0.45 on the same recent window) — its
apparent edge isn't statistically distinguishable from noise given the sample size. This
matches the expected outcome flagged in the plan: short-term daily reversal in SPY is a
decayed, largely post-2013-dead anomaly.

## Pivot tests (`scripts/run_pivots.py`)

After the base rule's NO-GO, the plan's documented pivots were run through the identical gate
(`meanrev/validation/gate.py`). **Result as of 2026-07-08: all six NO-GO.**

| Variant | Recent OOS Sharpe | p-value | Verdict |
|---|---|---|---|
| SPY daily fade, high-vol regime only (long/short) | -0.02 | 0.21 | NO-GO |
| SPY daily fade, high-vol regime only (long-only) | 0.82 | 0.45 | NO-GO |
| SPY weekly fade (long/short) | 0.15 | 0.06 | NO-GO |
| SPY weekly fade (long-only) | 1.19 | 0.23 | NO-GO |
| BTC-USD daily fade | -0.66 | 0.77 | NO-GO |
| ETH-USD daily fade | -0.03 | 0.28 | NO-GO |

The long-only variants show healthy-looking Sharpes but fail significance: a mostly-long
signal earns the equity/crypto risk premium regardless of *when* it's long, and the permutation
test (which preserves the long share while destroying the timing) shows the fade timing adds
nothing distinguishable from noise. The crypto results confirm the daily-reversal effect that
motivated the original notebook has decayed in that market as well.

## Tests

```
pytest
```
