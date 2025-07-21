import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np
import torch
from torch import Tensor


@dataclass
class Seeds:
    python: Tuple | List
    numpy: Tuple | Dict
    torch: Tensor | List[int]
    torch_cuda: Optional[Iterable[Tensor]]

    @classmethod
    def state_dict(cls) -> Dict:
        py = py_state_to_json()
        d_np = np_state_to_json()
        d_torch = torch_state_to_json()
        d_torch_cuda = None
        if torch.cuda.is_available():
            d_torch_cuda = cuda_state_to_json()
        seeds = Seeds(python=py, numpy=d_np, torch=d_torch, torch_cuda=d_torch_cuda)
        d = asdict(seeds)
        return d

    @classmethod
    def load_state_dict(cls, d: Dict):
        seeds = Seeds(**d)
        py = py_state_from_json(seeds.python)
        random.setstate(py)
        np_state = np_state_from_json(seeds.numpy)
        np.random.set_state(np_state)
        torch_state = torch_state_from_json(seeds.torch)
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and seeds.torch_cuda is not None:
            cuda_state = cuda_state_from_json(seeds.torch_cuda)
            torch.cuda.set_rng_state_all(cuda_state)


def py_state_to_json():
    state = random.getstate()
    version, keys, gauss_next = state
    return [version, list(keys), gauss_next]

def py_state_from_json(state: List) -> Tuple:
    version, keys, gauss_next = state
    return version, tuple(keys), gauss_next


def np_state_to_json():
    state = np.random.get_state()
    name, keys, pos, has_gauss, cached_gauss = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gauss": cached_gauss,
    }

def np_state_from_json(state: Dict) -> Tuple:
    name = state["name"]
    keys = np.array(state["keys"])
    pos = state["pos"]
    has_gauss = state["has_gauss"]
    cached_gauss = state["cached_gauss"]
    return name, keys, pos, has_gauss, cached_gauss


def torch_state_to_json() -> List[int]:
    return torch.get_rng_state().cpu().numpy().tolist()


def torch_state_from_json(state: List[int]) -> Tensor:
    return torch.tensor(state, dtype=torch.uint8)


def cuda_state_to_json() -> Optional[List[int]]:
    if not torch.cuda.is_available():
        return None
    return [torch.cuda.get_rng_state(i).cpu().numpy().tolist() for i in range(torch.cuda.device_count())]

def cuda_state_from_json(state: List[List[int]]) -> List[Tensor]:
    if not torch.cuda.is_available():
        return []
    return [torch.tensor(s, dtype=torch.uint8) for s in state]