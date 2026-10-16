"""Runs the finetunability signal for one checkpoint: a light, direct LoRA finetune plus a
held-out score per dataset, over TabArena's datasets. Each dataset is scored with its own TabArena
eval_metric (roc_auc for binary, log_loss for multiclass, rmse for regression) rather than one
metric for everything, and reported as three separate means — never averaged into one number,
since they're on different scales and directions. Every individual dataset's score is returned
too (`dataset_<name>` keys), for per-dataset analysis later, not just the aggregate.

A comparable number is instead produced via Elo (`finetunability/elo.py`): the
checkpoint's per-dataset scores are battled, metric-appropriately, against `BASELINE_SCORES_CSV`'s
TabArena-lite baselines scored under this same protocol -- separately for classification
(`cls_elo`) and regression (`reg_elo`) datasets (multiclass and binary are not split further, both
are still "classification" for this purpose), and once more across every dataset regardless of
task type (`elo`). `elo` is its own battle, not an average of `cls_elo`/`reg_elo`.

Requires `tabarena` and `bencheval` installed in this environment.
"""
import gc
import json
import subprocess
import sys
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Optional

import torch

from tabstarv1p1_pretrain.finetunability.datasets import FinetunabilityDatasets, select_finetunability_datasets
from tabstarv1p1_pretrain.finetunability.elo import CHECKPOINT_METHOD, compute_finetunability_elo, load_baseline_errors_csv, metric_error
from tabstarv1p1_pretrain.pretraining.pretrain_args import TaskType

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCORES_CSV = Path(__file__).resolve().parent / "baseline_scores.csv"
# A plain, always-present baseline (every dataset, no GPU-width failures) to hold fixed as the
# Elo zero point, so ratings stay comparable across separate calls even as every other score moves.
CALIBRATION_METHOD = "LightGBM"
TABARENA_TOOL_PACKAGES = ("tabarena", "bencheval")


def assert_finetunability_installed() -> None:
    """Called once before a pretrain run starts, not inside the per-epoch subprocess: a missing
    package there is silently caught (see `run_finetunability_check`'s docstring) so it would
    otherwise burn a full run with no finetunability signal at all before anyone notices.
    """
    missing = [pkg for pkg in TABARENA_TOOL_PACKAGES if find_spec(pkg) is None]
    if not missing:
        return
    raise RuntimeError(
        f"Pretraining requires {' and '.join(missing)} installed for finetunability tracking: "
        f"install `tabarena[benchmark]` and `bencheval` from the TabArena repo "
        f"(https://github.com/autogluon/tabarena)"
    )


def run_finetunability_check(checkpoint_dir: str, run_name: str, task_type: TaskType,
                             timeout_s: float = 1800.0,
                             raw_scores_dir: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Mean roc_auc / log_loss / rmse for one checkpoint over the full dataset sample, each
    averaged only over the datasets that actually use that metric. Runs
    `tabstarv1p1_pretrain.do_finetunability_check` as a subprocess, not in-process — the base estimator fixes the
    RNG seed unconditionally, which would perturb a live trainer's own RNG stream, and a second
    full model plus optimizer on the same GPU would have no crash isolation from the caller.

    `task_type` "cls"/"reg" drops the other task's datasets entirely before the subprocess is
    even launched, since a pretrain run restricted to one task type has no use for a finetunability
    signal on the other.

    A dataset that couldn't be finetuned is dropped (logged, not raised) rather than failing the
    whole call — a partial signal beats none, as long as dropout isn't too heavy (checked below).

    If `raw_scores_dir` is given, the per-dataset scores are copied there as
    `<run_name>.json` before `checkpoint_dir` (an ephemeral snapshot, see callers) is torn down —
    otherwise only the aggregate means below survive the call.

    Never raises: any failure (subprocess error, timeout, too few results) is logged and returns
    None, so a finetunability-check hiccup can't take down the training run that called it.

    The caller's own model keeps living on the GPU throughout, so before spawning the subprocess
    we release the caller's cached-but-unallocated CUDA blocks back to the driver — otherwise a
    training run deep enough to have grown its allocator footprint can starve the subprocess of
    room to fit its own model and batches, even though most of that footprint isn't in live use.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    try:
        datasets = select_finetunability_datasets()
        if task_type == TaskType.CLS:
            datasets = FinetunabilityDatasets(cls=datasets.cls, reg=[])
        elif task_type == TaskType.REG:
            datasets = FinetunabilityDatasets(cls=[], reg=datasets.reg)
        results_json = Path(checkpoint_dir) / "finetunability_check_results.json"
        cmd = [sys.executable, "-m", "tabstarv1p1_pretrain.do_finetunability_check",
               "--pretrain_dataset_or_path", checkpoint_dir, "--dataset_names", *datasets.all,
               "--results_json", str(results_json)]
        proc = subprocess.run(cmd, timeout=timeout_s, cwd=str(REPO_ROOT), capture_output=True, text=True)
        log_path = _save_subprocess_log(raw_scores_dir, run_name, proc)
        if proc.returncode != 0:
            raise RuntimeError(f"do_finetunability_check exited {proc.returncode}; see {log_path}")
        scores = json.loads(results_json.read_text())
        if raw_scores_dir is not None:
            Path(raw_scores_dir).mkdir(parents=True, exist_ok=True)
            (Path(raw_scores_dir) / f"{run_name}.json").write_text(json.dumps(scores))
        min_datasets = max(1, len(datasets.all) // 2)  # tolerate some dropout, not most of it
        if len(scores) < min_datasets:
            raise ValueError(f"Too few successful datasets to trust an average: "
                              f"{len(scores)}/{len(datasets.all)}; see {log_path}")
        by_metric = defaultdict(list)
        for dataset_result in scores.values():
            by_metric[dataset_result["eval_metric"]].append(dataset_result["score"])
        result = {f"mean_{metric}": sum(vals) / len(vals) for metric, vals in by_metric.items()}
        result.update({f"dataset_{name}": dataset_result["score"] for name, dataset_result in scores.items()})
        result.update(_compute_elo(scores, datasets))
        return result
    except Exception as e:
        print(f"⚠️ Finetunability check failed for {run_name}: {type(e).__name__}: {e}")
        return None


def _compute_elo(scores: Dict[str, Dict], datasets: FinetunabilityDatasets) -> Dict[str, float]:
    """`cls_elo` / `reg_elo`: the checkpoint's per-dataset scores battled against
    `BASELINE_SCORES_CSV`'s baselines, one battle per task type, restricted to whichever of
    that type's datasets `scores` actually covers. `elo` is a separate, single battle over every
    dataset `scores` covers regardless of task type -- not an average of `cls_elo`/`reg_elo`, since
    those are two independently-calibrated battles and averaging their outputs isn't a meaningful
    Elo. A task type absent from `scores` is skipped, not zeroed. Raises on a real failure
    (assert_finetunability_installed already guarantees bencheval is present) -- the caller's own
    try/except is what protects the training run.
    """
    checkpoint_errors = {name: metric_error(r["score"], r["eval_metric"]) for name, r in scores.items()}
    baseline_errors = load_baseline_errors_csv(str(BASELINE_SCORES_CSV))
    result = {}
    for key, task_datasets in [("cls_elo", datasets.cls), ("reg_elo", datasets.reg), ("elo", datasets.all)]:
        battle_datasets = [d for d in task_datasets if d in checkpoint_errors]
        if not battle_datasets:
            continue
        elo = compute_finetunability_elo(checkpoint_errors=checkpoint_errors, baseline_errors=baseline_errors,
                                         calibration_method=CALIBRATION_METHOD, datasets=battle_datasets)
        result[key] = elo[CHECKPOINT_METHOD]
    return result


def _save_subprocess_log(raw_scores_dir: Optional[str], run_name: str,
                         proc: subprocess.CompletedProcess) -> str:
    """The finetunability-check subprocess's full stdout/stderr, saved to its own file rather than
    embedded in a caller-facing failure message -- a stuck or OOM-ing dataset there can print far
    more than belongs inline in the main training log. Returns a path the failure message can
    point to instead.
    """
    if raw_scores_dir is None:
        return "<no raw_scores_dir given, log not saved>"
    log_dir = Path(raw_scores_dir) / "subprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    log_path.write_text(f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n")
    return str(log_path)
