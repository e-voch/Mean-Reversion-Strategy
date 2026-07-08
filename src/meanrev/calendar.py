"""NYSE trading calendar helpers. All date arithmetic in this project should go through here
rather than raw timedelta math, since equities don't trade on weekends/holidays like crypto does.
"""

import pandas as pd
import pandas_market_calendars as mcal

ANNUALIZATION_FACTOR = 252

_NYSE = mcal.get_calendar("NYSE")
_LOOKAHEAD = pd.Timedelta(days=14)


def sessions(start, end):
    """Trading session dates (normalized, tz-naive) between start and end, inclusive."""
    schedule = _NYSE.schedule(start_date=start, end_date=end)
    return schedule.index.tz_localize(None).normalize()


def is_session(date) -> bool:
    day = _to_naive_date(date)
    return day in sessions(day, day)


def is_early_close(date) -> bool:
    """True if the session on this date closes before the standard 16:00 ET close."""
    day = _to_naive_date(date)
    schedule = _NYSE.schedule(start_date=day, end_date=day)
    if schedule.empty:
        return False
    close = schedule.iloc[0]["market_close"].tz_convert("America/New_York")
    return close.hour < 16


def next_session(date):
    day = _to_naive_date(date)
    sched = _NYSE.schedule(start_date=day, end_date=day + _LOOKAHEAD)
    future = sched.index.tz_localize(None).normalize()
    future = future[future > day]
    return future[0]


def prev_session(date):
    day = _to_naive_date(date)
    sched = _NYSE.schedule(start_date=day - _LOOKAHEAD, end_date=day)
    past = sched.index.tz_localize(None).normalize()
    past = past[past < day]
    return past[-1]


def _to_naive_date(date):
    ts = pd.Timestamp(date)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()
