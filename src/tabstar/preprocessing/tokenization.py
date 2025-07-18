from typing import List

from transformers import AutoTokenizer, BatchEncoding

from tabstar.arch.config import E5_SMALL

TOKENIZER = {}


def tokenize(texts: List[str]) -> BatchEncoding:
    tokenizer = get_tokenizer()
    inputs = tokenizer(texts, padding=True, return_tensors='pt', truncation=True)
    return inputs


def get_tokenizer():
    if 'tokenizer' not in TOKENIZER:
        TOKENIZER['tokenizer'] = AutoTokenizer.from_pretrained(E5_SMALL)
    return TOKENIZER['tokenizer']

