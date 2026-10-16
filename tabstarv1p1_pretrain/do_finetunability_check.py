"""Finetunes a TabSTAR-v1.1 checkpoint directly on a subsample of TabArena's official splits for a set of
datasets, scoring each with its own TabArena eval_metric (roc_auc for binary, log_loss for
multiclass, rmse for regression). A checkpoint-level, cross-run-comparable proxy, not an absolute
"how good" score: same fixed datasets, same fixed protocol every call. Uses the exact same
finetuning hyperparameters as a real downstream finetune (`tabstar.tabstarv1p1.finetune.hyperparams`) --
the only thing making this "lite" is the train/test cap (`FINETUNABILITY_CHECK_MAX_TRAIN_SIZE` /
`FINETUNABILITY_CHECK_MAX_TEST_SIZE`), on TabArena's fold 0. At most one base model is loaded per
task type encountered, reused across every dataset of that type rather than reloaded from disk
each time. `--pretrain_dataset_or_path` is optional; omitting it defaults to the production SOTA
checkpoint for each dataset's task type (see `tabstar.tabstarv1p1.tabstar_checkpoints`).

Requires `tabarena` installed in this environment.

    python -m tabstarv1p1_pretrain.do_finetunability_check --dataset_names a b c --results_json out.json

A dataset that fails to finetune is skipped (logged, not raised) — the caller may still get a
usable signal from whichever datasets succeeded.
"""
import argparse
import json
import shutil
from typing import Dict, List, Optional, Union

from tqdm import tqdm

FINETUNABILITY_CHECK_MAX_TRAIN_SIZE = 1000
FINETUNABILITY_CHECK_MAX_TEST_SIZE = 500


def score_tabarena_metric(model, x_test, y_test, eval_metric: str) -> float:
    from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error

    if eval_metric == "rmse":
        return root_mean_squared_error(y_test, model.predict(x_test))
    y_true = model.preprocessor_.transform_target(y_test)
    y_pred = model.predict_proba(x_test)
    if eval_metric == "roc_auc":
        return roc_auc_score(y_true, y_pred[:, 1])
    if eval_metric == "log_loss":
        return log_loss(y_true, y_pred, labels=list(range(model.preprocessor_.d_output)))
    raise ValueError(f"Unsupported eval_metric: {eval_metric}")


def finetune_and_score_one(dataset_name: str, pretrain_dataset_or_path: str, base_model) -> Dict[str, Union[float, str]]:
    from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str
    from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
    from tabarena.contexts import TabArenaContext

    from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier, TabSTARv1p1Regressor

    metadata = TabArenaContext().task_metadata_collection.task_metadata_by_dataset()[dataset_name]
    tid = tid_from_task_id_str(metadata.task_id_str)
    task = OpenMLTaskWrapper.from_task_id(tid, metadata=metadata)
    x_train, y_train, x_test, y_test = task.get_train_test_split(
        fold=0, train_size=FINETUNABILITY_CHECK_MAX_TRAIN_SIZE, test_size=FINETUNABILITY_CHECK_MAX_TEST_SIZE)

    model_cls = TabSTARv1p1Classifier if metadata.is_classification else TabSTARv1p1Regressor
    # No hyperparameters passed: the estimator defaults are the downstream finetune ones.
    model = model_cls(pretrain_dataset_or_path=pretrain_dataset_or_path, preloaded_base_model=base_model,
                      device="cuda")
    model.fit(x_train, y_train)
    try:
        score = score_tabarena_metric(model, x_test, y_test, metadata.eval_metric)
        return {"score": float(score), "eval_metric": metadata.eval_metric}
    finally:
        # model.save_dir is scratch space for this one finetune; nothing reads it after scoring.
        if model.save_dir is not None:
            shutil.rmtree(model.save_dir, ignore_errors=True)


def main(pretrain_dataset_or_path: Optional[str], dataset_names: List[str], results_json: str) -> None:
    from tabarena.contexts import TabArenaContext

    from tabstar.tabstarv1p1.finetune.lora import load_pretrained_base
    from tabstar.tabstarv1p1.tabstar_checkpoints import get_tabstar_version

    task_metadata_by_dataset = TabArenaContext().task_metadata_collection.task_metadata_by_dataset()
    base_models = {}
    scores: Dict[str, Dict[str, Union[float, str]]] = {}
    for dataset_name in tqdm(dataset_names, desc="Finetunability"):
        try:
            is_cls = task_metadata_by_dataset[dataset_name].is_classification
            resolved_path = get_tabstar_version(pretrain_dataset_or_path=pretrain_dataset_or_path, is_cls=is_cls)
            if is_cls not in base_models:
                base_models[is_cls] = load_pretrained_base(resolved_path)
            scores[dataset_name] = finetune_and_score_one(dataset_name, resolved_path, base_models[is_cls])
        except Exception as e:
            print(f"⚠️ {dataset_name} failed: {type(e).__name__}: {e}")
    with open(results_json, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dataset_or_path", default=None,
                        help="Path to a pretrain-format checkpoint (see tabstar.tabstarv1p1.finetune.lora.load_pretrained_base). "
                             "Defaults to the production SOTA checkpoint for each dataset's task type.")
    parser.add_argument("--dataset_names", nargs="+", required=True, help="TabArena dataset names to finetune on.")
    parser.add_argument("--results_json", required=True,
                        help="Path to write {dataset_name: {score, eval_metric}} as JSON.")
    args = parser.parse_args()
    main(pretrain_dataset_or_path=args.pretrain_dataset_or_path, dataset_names=args.dataset_names,
        results_json=args.results_json)
