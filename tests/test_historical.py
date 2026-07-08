import pandas as pd
import pytest

from meanrev.data import historical


def _valid_df():
    sessions = historical.cal.sessions("2024-01-02", "2024-01-31")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "raw_close": 100.5,
            "volume": 1_000_000,
        },
        index=sessions,
    )


def test_quality_check_passes_clean_data():
    df = _valid_df()
    checked = historical._quality_check(df, "TEST")
    pd.testing.assert_frame_equal(checked, df)


def test_quality_check_rejects_nans():
    df = _valid_df()
    df.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        historical._quality_check(df, "TEST")


def test_quality_check_rejects_non_positive_prices():
    df = _valid_df()
    df.iloc[0, df.columns.get_loc("close")] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        historical._quality_check(df, "TEST")


def test_quality_check_rejects_missing_session():
    df = _valid_df()
    df = df.drop(df.index[5])
    with pytest.raises(ValueError, match="missing"):
        historical._quality_check(df, "TEST")


def test_quality_check_rejects_weekend_row():
    df = _valid_df()
    extra_row = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "raw_close": 100.5, "volume": 1},
        index=[pd.Timestamp("2024-01-06")],  # a Saturday
    )
    df = pd.concat([df, extra_row]).sort_index()
    with pytest.raises(ValueError, match="non-NYSE-session"):
        historical._quality_check(df, "TEST")


def _fake_download(symbol, start, end):
    sessions = historical.cal.sessions(start, end)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "raw_close": 100.5, "volume": 1_000},
        index=sessions,
    )


def test_load_reuses_cache_when_range_is_covered(monkeypatch, tmp_path):
    monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(historical, "_download", lambda *a: calls.append(a) or _fake_download(*a))

    historical.load("TEST", start="2024-01-02", end="2024-03-28")
    historical.load("TEST", start="2024-02-01", end="2024-03-01")  # narrower, already covered
    assert len(calls) == 1


def test_load_redownloads_when_requested_range_exceeds_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(historical, "_download", lambda *a: calls.append(a) or _fake_download(*a))

    historical.load("TEST", start="2024-01-02", end="2024-03-28")
    historical.load("TEST", start="2020-01-01", end="2024-03-28")  # earlier start, not covered
    assert len(calls) == 2
