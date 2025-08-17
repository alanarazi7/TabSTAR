import os
from typing import List

from transformers import AutoTokenizer, BatchEncoding

from tabstar.arch.config import E5_SMALL

TOKENIZER = {}


def tokenize(texts: List[str]) -> BatchEncoding:
    tokenizer = _get_tokenizer_for_worker()
    inputs = tokenizer(texts, padding=True, return_tensors='pt', truncation=True)
    return inputs


def _get_tokenizer_for_worker():
    pid = os.getpid()
    if pid not in TOKENIZER:
        TOKENIZER[pid] = AutoTokenizer.from_pretrained(E5_SMALL)
    return TOKENIZER[pid]