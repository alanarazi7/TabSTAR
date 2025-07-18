from tabular.constants import VERBOSE


def verbose_print(s: str):
    # TODO: use logger
    if VERBOSE:
        print(s)
