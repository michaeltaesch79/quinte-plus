"""Paris local time without a tz database (the Railway image has none)."""
from datetime import date, datetime, timedelta, timezone


def last_sunday(year, month):
    d = date(year, month, 31)  # March and October both have 31 days
    return d - timedelta(days=(d.weekday() + 1) % 7)


def paris_now():
    """EU summer time: from the last Sunday of March to the last Sunday of October, 01:00 UTC."""
    utc = datetime.now(timezone.utc)
    start = datetime.combine(last_sunday(utc.year, 3), datetime.min.time(), timezone.utc) + timedelta(hours=1)
    end = datetime.combine(last_sunday(utc.year, 10), datetime.min.time(), timezone.utc) + timedelta(hours=1)
    return (utc + timedelta(hours=2 if start <= utc < end else 1)).replace(tzinfo=None)
