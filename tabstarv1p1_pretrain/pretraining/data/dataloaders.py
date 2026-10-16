from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tabstar.tabstarv1p1.preprocessing.tokenization import tokenize_x_txt
from tabstarv1p1_pretrain.pretraining.arguments import D_MODEL
from tabstarv1p1_pretrain.pretraining.hdf5 import RealDatasetProperties
from tabstar.tabstarv1p1.training.batching import TabSTARBatch


def pretrain_batch_collator(examples) -> TabSTARBatch:
    x_num, x_txt, y, properties = zip(*examples)
    x_num, feature_mask = consolidate_x_num(x_num)
    x_txt, tokenized = tokenize_x_txt(x_txt=x_txt)
    if all(isinstance(p, RealDatasetProperties) for p in properties):
        d_outputs = torch.tensor([p.d_output for p in properties], dtype=torch.long)
    elif all(isinstance(p, int) for p in properties):
        assert len(set(properties)) == 1
        d_outputs = torch.tensor(properties, dtype=torch.long)
    else:
        raise ValueError("Properties must be all RealDatasetProperties or all int!")
    y, is_cls = consolidate_y(y, d_outputs=d_outputs)
    batch = TabSTARBatch(x_num=x_num, x_txt=x_txt, tokenized=tokenized, y=y, d_outputs=d_outputs,
                          feature_mask=feature_mask, is_cls=is_cls)
    return batch

def consolidate_x_num(x_num: List[np.ndarray]) -> Tuple[Tensor, Tensor]:
    seq_tensors = [torch.as_tensor(x, dtype=torch.bfloat16) for x in x_num]
    x_num = pad_sequence(seq_tensors, batch_first=True, padding_value=float("nan"))
    feature_mask = torch.isnan(x_num)
    assert feature_mask.dim() == 3
    feature_mask = feature_mask.any(dim=-1)
    assert feature_mask.dim() == 2
    x_num = torch.nan_to_num(x_num, nan=0.0)
    return x_num, feature_mask

def consolidate_y(y: List[np.ndarray | int], d_outputs: Tensor) -> Tuple[Tensor, bool]:
    if max(d_outputs) == 1:
        dtype = torch.bfloat16
        is_cls = False
    elif min(d_outputs) >= 2:
        dtype = torch.long
        is_cls = True
    else:
        raise NotImplementedError("We don't support mixed classification and regression batches!")
    y = torch.tensor(y)
    y = y.to(dtype=dtype)
    return y, is_cls


def consolidate_x_txt(x_txt: List[np.ndarray]) -> Tensor:
    assert all(arr.shape[1] == D_MODEL and arr.ndim == 2 for arr in x_txt)
    max_cols = max(arr.shape[0] for arr in x_txt)
    batch_size = len(x_txt)
    x_txt = [torch.tensor(arr, dtype=torch.bfloat16) for arr in x_txt]
    x_txt = pad_sequence(x_txt, batch_first=True, padding_value=0.0)
    assert x_txt.shape == (batch_size, max_cols, D_MODEL)
    return x_txt
