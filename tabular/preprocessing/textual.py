import re


def sanitize_text(text: str) -> str:
    # Remove control characters except for standard whitespace (newline, tab, etc.)
    return re.sub(r'[\x00-\x1F\x7F]', ' ', text)


def replace_unspaced_symbols(text: str) -> str:
    if ' ' not in text:
        return text
    for c in ['_', '-', "."]:
        text = text.replace(c, ' ')
    return text


def normalize_col_name(text: str) -> str:
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\u00A0', ' ')
    text = replace_unspaced_symbols(text)
    return text
