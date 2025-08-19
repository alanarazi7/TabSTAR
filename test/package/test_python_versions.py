def test_python_version():
    """
    match doesn't exists in 3.9.
    this tests tries to use match.
    if it's 3.11, it returns False to fail 
    """
    value_1 = 42
    value_2 = 42
    try: 
        match value_1:
            case value_2:
                assert False # this code runs only if value_1 and value_2 match
    except SyntaxError:
        assert True # passes in 3.9 
