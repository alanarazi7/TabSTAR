def test_python_version():
    """
    should raise a syntax error in 3.9
    should pass in 3.11
    
    """
    value_1 = 42
    value_2 = 42
    match value_1:
        case value_2:
            assert True
        