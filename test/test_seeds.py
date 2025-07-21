from tabstar.training.utils import fix_seed
from tabstar_paper.pretraining.seeds import py_state_to_json, py_state_from_json, np_state_to_json, np_state_from_json, \
    torch_state_to_json, torch_state_from_json


def test_py_seeds():
    fix_seed(0)
    py = py_state_to_json()
    py = py_state_from_json(py)
    assert py[0] == 3
    assert py[1][:5] == (2147483648, 766982754, 497961170, 3952298588, 2331775348)
    assert py[2] is None

def test_np_seeds():
    fix_seed(0)
    np_state = np_state_to_json()
    np_state = np_state_from_json(np_state)
    assert np_state[0] == 'MT19937'
    assert np_state[1].shape == (624,)
    assert np_state[1].tolist()[:5] == [0, 1, 1812433255, 1900727105, 1208447044]
    assert np_state[2] == 624
    assert np_state[3] == 0
    assert np_state[4] == 0.0


def test_torch_seeds():
    fix_seed(1306)
    torch_state = torch_state_to_json()
    torch_state = torch_state_from_json(torch_state)
    assert torch_state.tolist()[:3] == [26, 5, 0]
