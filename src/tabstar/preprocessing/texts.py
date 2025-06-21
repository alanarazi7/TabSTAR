import re
from typing import Tuple, Optional

from pandas import DataFrame, Series


def replace_column_names(x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
    old2new = {c: normalize_col_name(c) for c in x.columns}
    x.rename(columns=old2new, inplace=True)
    if y is not None:
        y.name = normalize_col_name(str(y.name))
    return x, y

def normalize_col_name(text: str) -> str:
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\u00A0', ' ')
    text = replace_unspaced_symbols(text)
    text = replace_whitespaces(text)
    return text

def replace_unspaced_symbols(text: str) -> str:
    if ' ' not in text:
        return text
    for c in ['_', '-', "."]:
        text = text.replace(c, ' ')
    return text

def replace_whitespaces(text: str) -> str:
    # TODO: originally this was supposed to keep newlines and tabs
    return re.sub(r'[\x00-\x1F\x7F]', ' ', text)
