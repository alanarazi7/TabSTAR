from typing import Tuple

from pandas import DataFrame, Series


def remove_semantics(x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    # Since the data-synth generator works only with numerical data, we remove the semantics
    raise NotImplementedError()

