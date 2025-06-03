from typing import List

import numpy as np
from torch import Tensor


def concat_predictions(y_pred: List[Tensor]) -> np.ndarray:
    return np.concatenate([p.cpu().detach().numpy() for p in y_pred])