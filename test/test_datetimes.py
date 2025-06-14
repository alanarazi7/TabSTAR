import pandas as pd
from pandas import Timestamp

from tabstar.preprocessing.dates import series_to_dt


def test_unix_timestamps():
    values = [1500000000, 999999999999990021]
    expected = ['1970-01-01 00:00:01.500000', '2001-09-09 01:46:39.999990021']
    _assert_expected(values, expected)

def test_day_only():
    values = ["2017-10-02"]
    expected = ["2017-10-02 00:00:00"]
    _assert_expected(values, expected)

def test_iso_without_timezone():
    values = ["2022-03-15T12:30:00"]
    expected = ["2022-03-15 12:30:00"]
    _assert_expected(values, expected)

def test_iso_with_timezone():
    values = ["2017-10-02 17:11:16+00"]
    expected = ["2017-10-02 17:11:16"]  # timezone info removed
    _assert_expected(values, expected)

def test_with_bad_quotes():
    values = ["8/3/2017", "4/6/2014"]
    expected = ["2017-08-03 00:00:00", "2014-04-06 00:00:00"]
    _assert_expected(values, expected)

def test_invalid_date():
    values = ["not a date", None]
    expected = [pd.NaT, pd.NaT]
    _assert_expected(values, expected)


def _assert_expected(values, expected):
    """
    Converts the input values using series_to_dt and compares each result
    with the expected values. Expected values can be either a datetime string
    (which is converted to a Timestamp) or pd.NaT for invalid dates.
    """
    results = list(series_to_dt(pd.Series(values)))
    for i, (res, exp) in enumerate(zip(results, expected)):
        if pd.isna(exp):
            assert pd.isna(res), f"Row {i}: expected NaT, got {res}"
        else:
            # Convert expected value to a Timestamp and compare
            assert res == Timestamp(exp), f"Row {i}: expected {Timestamp(exp)}, got {res}"
