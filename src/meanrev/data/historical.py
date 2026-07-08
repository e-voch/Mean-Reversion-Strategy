"""Historical daily OHLCV data for research/backtesting, via yfinance with a local parquet cache."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from meanrev import calendar as cal

CACHE_DIR = Path(__file__).resolve().parents[3] / "data"


def load(symbol: str, start: str = "1990-01-01", end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Adjusted + raw daily OHLCV for `symbol`, cached to parquet under data/.

    Columns: open, high, low, close (adjusted), raw_close (unadjusted), volume.
    Adjusted close is what return/signal calculations must use — SPY pays dividends,
    and raw close shows artificial jumps on ex-dividend dates that would corrupt log returns.
    """
    cache_path = CACHE_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(cache_path) if cache_path.exists() and not refresh else None

    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    if df is None or df.index.min() > requested_start or df.index.max() < requested_end:
        df = _download(symbol, start, end)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return _quality_check(df, symbol)


def _download(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    adjusted = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if adjusted.empty or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} — check connectivity/date range")

    adjusted.columns = adjusted.columns.get_level_values(0)
    raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=adjusted.index)
    df["open"] = adjusted["Open"]
    df["high"] = adjusted["High"]
    df["low"] = adjusted["Low"]
    df["close"] = adjusted["Close"]
    df["raw_close"] = raw["Close"]
    df["volume"] = adjusted["Volume"]
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    return df


def _quality_check(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.isna().any().any():
        raise ValueError(f"{symbol}: NaNs present in cached data — refresh or investigate")
    if (df[["open", "high", "low", "close", "raw_close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive prices present in cached data")

    expected = cal.sessions(df.index.min(), df.index.max())
    missing = expected.difference(df.index)
    extra = df.index.difference(expected)
    if len(missing) > 0:
        raise ValueError(f"{symbol}: {len(missing)} expected NYSE sessions missing from data, e.g. {missing[:5].tolist()}")
    if len(extra) > 0:
        raise ValueError(f"{symbol}: {len(extra)} rows fall on non-NYSE-session dates, e.g. {extra[:5].tolist()}")

    return df
