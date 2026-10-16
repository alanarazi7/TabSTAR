from typing import Iterable, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from tabstar.tabstarv1p1.constants import TEXT_MODEL

_TOKENIZER_CACHE = {}

def get_tokenizer(model_name: str = TEXT_MODEL):
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return _TOKENIZER_CACHE[model_name]


def tokenize_x_txt(x_txt: Iterable[np.ndarray]) -> Tuple[Tensor, BatchEncoding]:
    tokenizer = get_tokenizer()
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    x_txt = _pad_x_txt(x_txt=x_txt)
    unique_texts, inv = map_unique_strings(x_txt=x_txt)
    assert inv.shape == x_txt.shape
    tokenized = tokenizer(unique_texts, padding=True, return_tensors='pt', truncation=True)
    x_txt_indices = torch.tensor(inv, dtype=torch.long)
    return x_txt_indices, tokenized


def _pad_x_txt(x_txt: Iterable[np.ndarray]) -> np.ndarray:
    batch_size = len(x_txt)
    rows = [list(r) for r in x_txt]
    max_len = max(len(r) for r in rows)
    to_pad_num = [max_len - len(r) for r in rows]
    padded = [[str(s) for s in r] + [""] * tp for r, tp in zip(rows, to_pad_num)]
    padded = np.array(padded)
    assert padded.shape == (batch_size, max_len), f"Got {padded.shape}, expected ({batch_size}, {max_len})"
    return padded


def map_unique_strings(x_txt: np.ndarray):
    uniq, inv = np.unique(x_txt, return_inverse=True)
    unique_texts = uniq.tolist()
    inv = inv.reshape(x_txt.shape)
    return unique_texts, inv


def trim_batch_to_max_size(tokenized: Dict[str, Tensor]) -> Dict[str, Tensor]:
    attn_mask = tokenized['attention_mask']
    max_len = attn_mask.sum(dim=1).max().item()
    return {k: v[:, :max_len] for k, v in tokenized.items()}
