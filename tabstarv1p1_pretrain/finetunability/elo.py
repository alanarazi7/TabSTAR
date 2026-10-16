"""Elo rating for the finetunability signal, battled against baseline models scored under the
same protocol (same fold, same train/test size, stored in `baseline_scores.csv`), not against a single anchor. Every
method present in `results` is rated together in one N-way battle; `bencheval.elo_utils.EloHelper`
already handles this natively once every method's scores are in one long-format table, so this
module is mostly that table's construction.

Both sides use TabArena's own per-dataset `eval_metric` (`roc_auc` for binary, `log_loss` for
multiclass, `rmse` for regression) — the metric the finetunability signal and the baseline scores in
`baseline_scores.csv` were computed with, so no separate scoring pass is needed here.
`metric_error()` turns that mix into one 'lower always wins' direction per row before the battle;
the three metrics are never combined into a shared scale, only compared dataset-by-dataset within
`EloHelper`'s own pairwise battles.

Uses the cheap, non-bootstrapped `compute_mle_elo`, appropriate for something meant to run every
epoch, not the bootstrapped, CI-bearing `compute_elo_ratings` a rarer periodic signal would afford.

Requires `bencheval` (from the TabArena repo) installed in this environment.
"""
from typing import Dict, List, Optional

import pandas as pd

CHECKPOINT_METHOD = "TabSTAR-v1.1-ckpt"


def metric_error(score: float, eval_metric: str) -> float:
    """`roc_auc` is higher-is-better and bounded above at 1; `log_loss` and `rmse` are already
    lower-is-better and unbounded, so they pass through unchanged.
    """
    if eval_metric == "roc_auc":
        return 1.0 - score
    if eval_metric in ("log_loss", "rmse"):
        return score
    raise ValueError(f"Unsupported eval_metric: {eval_metric}")


def load_baseline_errors_csv(path: str) -> pd.DataFrame:
    """Loads a baseline-scores CSV (columns: method, dataset, score, eval_metric) into the
    long-format [method, dataset, metric_error] frame `compute_finetunability_elo` expects.
    """
    df = pd.read_csv(path)
    df["metric_error"] = df.apply(lambda row: metric_error(row["score"], row["eval_metric"]), axis=1)
    return df[["method", "dataset", "metric_error"]]


def compute_finetunability_elo(checkpoint_errors: Dict[str, float], baseline_errors: pd.DataFrame,
                               calibration_method: str, calibration_elo: float = 1000.0,
                               datasets: Optional[List[str]] = None) -> Dict[str, float]:
    """Elo rating for every method present, rated together in one battle.

    `checkpoint_errors` maps dataset name -> `metric_error` for the checkpoint being scored.
    `baseline_errors` is long-format: one row per (method, dataset) with columns `method`,
    `dataset`, `metric_error` (e.g. from `load_baseline_errors_csv`). A method only needs scores
    on the datasets it shares with others, not every dataset.

    `datasets`, if given, restricts the battle to that subset (e.g. classification-only or
    regression-only datasets) instead of everything both sides cover.

    `calibration_method` is held fixed at `calibration_elo` in every call, so ratings stay
    comparable across separate calls (e.g. one per checkpoint/epoch) even as every other score
    moves — pick a method with stable, already-trusted scores, not the checkpoint itself.
    """
    from bencheval.elo_utils import EloHelper

    checkpoint_rows = [{"method": CHECKPOINT_METHOD, "dataset": d, "metric_error": e}
                       for d, e in checkpoint_errors.items()]
    results_df = pd.concat([pd.DataFrame(checkpoint_rows),
                            baseline_errors[["method", "dataset", "metric_error"]]], ignore_index=True)

    helper = EloHelper(method_col="method", task_col="dataset", error_col="metric_error")
    battles = helper.convert_results_to_battles(results_df, datasets=datasets)
    elo = helper.compute_mle_elo(battles, calibration_framework=calibration_method, calibration_elo=calibration_elo)
    return elo.to_dict()
