from dataclasses import dataclass

import numpy as np


@dataclass
class Predictions:
    score: float
    predictions: np.ndarray
    labels: np.ndarray