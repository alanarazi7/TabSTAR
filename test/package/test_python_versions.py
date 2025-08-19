
def test_python_version_functionality():
    """
    The "match" syntax was introduced in 3.10. So this should fail in 3.9, and pass in 3.11
    """
    match x:
        case 1:
            pass