import re
from typing import Any


def sanitize_text(text: str) -> str:
    # Remove control characters except for standard whitespace (newline, tab, etc.)
    return re.sub(r'[\x00-\x1F\x7F]', ' ', text)


def replace_unspaced_symbols(text: str) -> str:
    if ' ' not in text:
        return text
    for c in ['_', '-', "."]:
        text = text.replace(c, ' ')
    return text


def remove_commas(text: str) -> str:
    return text.replace(',', '')

def remove_percentage(text: Any) -> float:
    if not isinstance(text, str):
        return text
    return float(text.replace('%', ''))

def remove_currency(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return float(text.replace('$', ''))

def normalize_col_name(text: str) -> str:
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\u00A0', ' ')
    text = replace_unspaced_symbols(text)
    return text


def convert_currency_k_m(text: str) -> float:
    text = text.lower()
    if text.endswith('k'):
        return float(text[:-1]) / 1000
    elif text.endswith('m'):
        return float(text[:-1])
    else:
        return float(text)

def convert_weight_lbs(text: str) -> float:
    return float(text.replace('lbs', ''))


def remove_k(text: str) -> float:
    if text.isdigit():
        return float(text)
    text = text.lower()
    assert text.endswith('k')
    return float(text[:-1]) * 1000