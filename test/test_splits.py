import pandas as pd

from tabstar.preprocessing.splits import merge_splits

# TODO: AI assisted tests, need to understand better if they make sense
def _make_toy(overlap_indices=False):
    # Build data where y equals row-wise sum of x, so we can verify alignment after shuffling
    x1 = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    x2 = pd.DataFrame({"a": [4, 5], "b": [40, 50]})
    if overlap_indices:
        # Force duplicate labels across splits (0,1) to simulate a common pitfall
        x2.index = x1.index[:len(x2)]
    y1 = x1.sum(axis=1)  # 11, 22, 33
    y2 = x2.sum(axis=1)  # 44, 55
    return x1, x2, y1, y2

def test_merge_splits_unique_indices_aligns():
    x1, x2, y1, y2 = _make_toy(overlap_indices=False)
    x, y = merge_splits(x1, x2, y1, y2, random_state=42)
    # Recompute the per-row sum and ensure it equals shuffled y
    recomputed = x.sum(axis=1).reset_index(drop=True)
    pd.testing.assert_series_equal(recomputed, y.reset_index(drop=True), check_names=False)

def test_merge_splits_overlapping_indices_aligns():
    x1, x2, y1, y2 = _make_toy(overlap_indices=True)
    x, y = merge_splits(x1, x2, y1, y2, random_state=42)
    recomputed = x.sum(axis=1).reset_index(drop=True)
    pd.testing.assert_series_equal(recomputed, y.reset_index(drop=True), check_names=False)
