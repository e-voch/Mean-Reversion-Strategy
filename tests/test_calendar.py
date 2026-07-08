from meanrev import calendar as cal


def test_holiday_is_not_a_session():
    assert cal.is_session("2024-01-01") is False  # New Year's Day


def test_weekday_is_a_session():
    assert cal.is_session("2024-01-02") is True


def test_next_session_skips_weekend():
    assert cal.next_session("2023-12-29") == cal._to_naive_date("2024-01-02")


def test_prev_session_skips_weekend():
    assert cal.prev_session("2024-01-02") == cal._to_naive_date("2023-12-29")


def test_day_after_thanksgiving_is_early_close():
    assert cal.is_early_close("2024-11-29") is True


def test_ordinary_day_is_not_early_close():
    assert cal.is_early_close("2024-01-02") is False


def test_annualization_factor_is_252_not_365():
    assert cal.ANNUALIZATION_FACTOR == 252
