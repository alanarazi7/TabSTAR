import re

def test_regex_match_truthiness():
    match = re.match(r"abc", "def")  # No match
    # In Python 3.9: match is falsy (None)
    # In Python 3.11: match is always truthy (if not None)
    assert not match