from tabstar.preprocessing.binning import format_float


def test_numerical_verbalizations():
    assert format_float(3.1415926) == "3.1416"
    assert format_float(0.0000001) == "0"
    assert format_float(0.00001) == "0"
    assert format_float(0.0500) == "0.05"