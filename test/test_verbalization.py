from tabstar.preprocessing.texts import replace_whitespaces
from tabstar.preprocessing.verbalize import verbalize_feature


def test_replace_whitespace():
    assert replace_whitespaces('a text\nwith newline') == 'a text with newline'
    assert replace_whitespaces('a text   with multiple spaces') == 'a text   with multiple spaces'


def test_verbalize_feature():
    f = verbalize_feature(col="f1", value="value1")
    assert f == "Predictive Feature: f1\nFeature Value: value1"