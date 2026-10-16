from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from pandas import DataFrame, Series


# TODO: this seems like a leftover
@dataclass
class TabSTARData:
    d_output: int
    x_txt: Union[DataFrame, np.ndarray]
    x_num: np.ndarray
    y: Optional[Series] = None
    x_embed: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.x_txt)
