from typing import List

import numpy as np


def prediction_spread(val_predictions: np.ndarray) -> float:
    """How much predictions vary across rows: the largest per-class std, or the std of a regression output."""
    per_output_std = np.std(val_predictions.reshape(len(val_predictions), -1), axis=0)
    return float(per_output_std.max())


def select_epochs(val_scores: List[float], spreads: List[float], k: int, collapse_ratio: float) -> List[int]:
    """The k epochs with the lowest val score, skipping collapsed ones: an epoch whose prediction spread fell
    below `collapse_ratio` of the best epoch's predicts near-constant output. Returns 1-based epochs."""
    scores = np.asarray(val_scores)
    spreads = np.asarray(spreads)
    by_score = np.argsort(scores, kind="stable")
    min_spread = spreads[by_score[0]] * collapse_ratio
    healthy = [int(i) + 1 for i in by_score if spreads[i] >= min_spread]
    return healthy[:k]
