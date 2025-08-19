from collections import OrderedDict

def test_dictionary_and_ordered_dictionary_equality():
    """
    This test demonstrates a change in Python's behavior between versions 3.9 and 3.11.
    In Python 3.9, a regular dictionary and an OrderedDict with the same contents are not considered equal.
    In Python 3.11, they are considered equal. Therefore, this test will pass in Python 3.9 but fail in Python 3.11.
    """
    regular_dictionary = {'key_one': 1, 'key_two': 2}
    ordered_dictionary = OrderedDict([('key_one', 1), ('key_two', 2)])
    # In Python 3.9: regular_dictionary != ordered_dictionary
    # In Python 3.11: regular_dictionary == ordered_dictionary if contents are the same
    assert regular_dictionary != ordered_dictionary