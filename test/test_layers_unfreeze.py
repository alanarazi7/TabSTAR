from tabular.utils.deep import get_last_layers_num


def test_unfreeze_layers():
    assert get_last_layers_num(total_layers=10, to_unfreeze=3) == [9, 8, 7]
    assert get_last_layers_num(total_layers=10, to_unfreeze=0) == []
    assert get_last_layers_num(total_layers=12, to_unfreeze=2) == [11, 10]
    assert get_last_layers_num(total_layers=12, to_unfreeze=12) == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]