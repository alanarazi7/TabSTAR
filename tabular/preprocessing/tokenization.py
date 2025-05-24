from typing import List, Dict

import torch
from tabular.tabstar.params.constants import E5_SMALL
from transformers import AutoTokenizer, BatchEncoding

TOKENIZER = {}


def tokenize(texts: List[str], device: torch.device) -> BatchEncoding | Dict:
    tokenizer = get_tokenizer()
    inputs = tokenizer(texts, padding=True, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def get_tokenizer():
    if 'tokenizer' not in TOKENIZER:
        TOKENIZER['tokenizer'] = AutoTokenizer.from_pretrained(E5_SMALL)
    return TOKENIZER['tokenizer']

