from typing import List

import numpy as np
from numpy.exceptions import AxisError
from sklearn.metrics import r2_score, roc_auc_score
import torch
from torch import softmax, Tensor
from torch.nn import CrossEntropyLoss, MSELoss

from tabstar.tabstarv1p1.constants import REG_Z_CLIP
from tabstar.tabstarv1p1.utils.tensors import to_numpy


def mask_predictions(predictions: Tensor, d_outputs: Tensor) -> Tensor:
    if max(d_outputs) == 1:
        # dim=1 (not a bare squeeze()) so a batch size of 1 doesn't also collapse the batch dim,
        # and so the dual regression head's trailing (1 + n_bins) dimension survives untouched.
        return predictions.squeeze(dim=1).float()
    elif min(d_outputs) == 1:
        raise NotImplementedError("Regression in mixed batches is not supported yet.")
    row_idx = torch.arange(predictions.size(1)).unsqueeze(0).to(predictions.device)
    cutoffs = d_outputs.unsqueeze(1)
    mask = (row_idx < cutoffs)
    min_val = torch.finfo(predictions.dtype).min
    masked_predictions = predictions.masked_fill(~mask, min_val)
    return masked_predictions


def _distributional_expectation(logits: Tensor, bin_centers: Tensor) -> Tensor:
    probs = softmax(logits, dim=-1)
    return (probs * bin_centers.to(probs.dtype)).sum(dim=-1)


def reg_predictions_to_scalar(predictions: Tensor, bin_centers: Tensor) -> Tensor:
    """The one place a raw dual-head regression prediction becomes a z-score."""
    scalar_part = predictions[..., 0]
    dist_expectation = _distributional_expectation(predictions[..., 1:], bin_centers=bin_centers)
    scalar = (scalar_part + dist_expectation) / 2
    return scalar.clamp(min=-REG_Z_CLIP, max=REG_Z_CLIP)


def process_predictions(prediction: Tensor, d_outputs: Tensor, bin_centers: Tensor = None) -> Tensor:
    if max(d_outputs) == 1:
        return reg_predictions_to_scalar(prediction, bin_centers=bin_centers)
    elif min(d_outputs) >= 2:
        return softmax(prediction, dim=1)
    raise NotImplementedError("We don't support mixed classification and regression just yet.")


def calculate_mixed_metric(y_true: Tensor | np.ndarray, y_pred: Tensor | np.ndarray, d_outputs: Tensor | List[int]) -> float:
    if isinstance(y_true, Tensor):
        y_true = to_numpy(y_true)
    if isinstance(y_pred, Tensor):
        y_pred = to_numpy(y_pred)
    if isinstance(d_outputs, Tensor):
        d_outputs = d_outputs.cpu().numpy().tolist()
    if max(d_outputs) == 1:
        # This calculation might be over normalized `y_true`, so not necessarily comparable
        return r2_score(y_true=y_true, y_pred=y_pred)
    elif min(d_outputs) >= 2:
        return multiclass_roc_auc_score(y_true=y_true, y_pred=y_pred)
    raise NotImplementedError("We don't support mixed classification and regression just yet.")


def multiclass_roc_auc_score(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        # sklearn's OVR macro AUC warns and returns nan for this rather than raising, for
        # binary d_output specifically, so it must be caught before the call, not after.
        return float("nan")
    try:
        return roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
    except (ValueError, AxisError):
        # Error calculating AUC, likely due to class imbalance or insufficient samples
        return _per_class_auc(y_true=y_true, y_pred=y_pred)


def _per_class_auc(y_true, y_pred) -> float:
    present_classes = np.unique(y_true)
    aucs = []
    for cls in present_classes:
        y_true_binary = (y_true == cls).astype(int)
        if len(np.unique(y_true_binary)) < 2:
            continue
        y_pred_scores = y_pred[:, int(cls)]
        aucs.append(roc_auc_score(y_true_binary, y_pred_scores))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


# TODO: this is O(n_bins^2) via the pairwise term below. An O(n_bins) equivalent (CDF/ranked-
# probability-score form, valid since bins are evenly spaced) is already written and benchmarked
# at ~31x faster fwd+bwd; see commit f3d5117e on this branch. Left unapplied here so the CRPS+CE
# blend can first be ablated against the code as actually run, before optimizing it.
def _categorical_crps(logits: Tensor, bin_centers: Tensor, y: Tensor) -> Tensor:
    """Closed-form CRPS (equivalently the beta=1 energy score) for a categorical distribution
    over fixed bin centers, avoiding a separate quantile-head representation.

    CRPS(P, y) = E_X~P[|X - y|] - 0.5 * E_X,X'~P[|X - X'|], for X, X' iid draws from P.
    """
    probs = softmax(logits, dim=-1)
    centers = bin_centers.to(probs.dtype)
    dist_to_y = (centers.unsqueeze(0) - y.unsqueeze(-1)).abs()
    expected_dist_to_y = (probs * dist_to_y).sum(dim=-1)
    center_pairwise_dist = (centers.unsqueeze(0) - centers.unsqueeze(1)).abs()
    expected_pairwise_dist = torch.einsum("ni,nj,ij->n", probs, probs, center_pairwise_dist)
    crps = expected_dist_to_y - 0.5 * expected_pairwise_dist
    return crps.mean()


def _distributional_loss(logits: Tensor, y: Tensor, bin_edges: Tensor, bin_centers: Tensor) -> Tensor:
    bin_idx = torch.bucketize(y, bin_edges[1:-1].to(y.dtype))
    ce_loss = CrossEntropyLoss()(input=logits, target=bin_idx)
    crps_loss = _categorical_crps(logits, bin_centers=bin_centers, y=y)
    return 0.5 * ce_loss + 0.5 * crps_loss


def calculate_mixed_loss(y: Tensor, predictions: Tensor, d_outputs: Tensor, bin_edges: Tensor = None,
                         bin_centers: Tensor = None) -> Tensor:
    if max(d_outputs) == 1:
        scalar_loss = MSELoss()(input=predictions[..., 0].reshape(-1), target=y)
        dist_loss = _distributional_loss(predictions[..., 1:], y, bin_edges=bin_edges, bin_centers=bin_centers)
        return 0.5 * scalar_loss + 0.5 * dist_loss
    elif min(d_outputs) >= 2:
        return CrossEntropyLoss()(input=predictions, target=y)
    raise NotImplementedError("We don't support mixed classification and regression just yet.")
