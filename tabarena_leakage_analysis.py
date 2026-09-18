"""Quantify TabSTAR's potential pretraining leakage on TabArena-Lite.

TabSTAR is pretrained on a large corpus of tabular datasets, five leave-one-fold-out
checkpoints (`TabSTAR-eval-320-version-fold-k{0..4}`), each excluding a different ~1/5 slice
of the corpus. A handful of TabArena datasets overlap with that pretraining corpus, so scoring
the base checkpoint on them risks leakage. This script quantifies that risk by running each
overlapping dataset three ways through TabArena-Lite:

  (a) base    — the public base checkpoint (`alana89/TabSTAR`), pretrained on everything.
  (b) correct — the fold checkpoint that EXCLUDED this dataset from pretraining (leakage-free).
  (c) wrong   — a different fold checkpoint that still INCLUDED this dataset (sanity control:
                if (a) beats (c) similarly to how it beats (b), the gap is not about leakage).

If leakage matters, (a) should score above (b), and (c) should track (a) rather than (b).

The dataset -> checkpoint-fold mapping reuses TabSTAR's own name matching (PR #23,
`tabstar.tabstar_datasets.PRETRAIN2FOLD` / `TEXT2FOLD`), which maps a TabArena dataset name to
the pretraining-corpus key that was excluded, and from there to a fold index.

Requires the `tabarena` package (`pip install tabarena`), which is not a declared dependency of
this repo.

Usage:
    python tabarena_leakage_analysis.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

if TYPE_CHECKING:
    import pandas as pd

PRETRAIN_FOLD_REPO_TEMPLATE = "alana89/TabSTAR-eval-320-version-fold-k{fold}"


@dataclass(frozen=True)
class OverlapDataset:
    """A TabArena dataset that overlaps with TabSTAR's pretraining corpus."""

    tabarena_name: str
    """Dataset name as registered in TabArena (`build_kwargs={"dataset_names": [...]}`)."""
    tabstar_key: str
    """The pretraining-corpus key this dataset maps to (`tabstar.tabstar_datasets.PRETRAIN2FOLD`)."""
    correct_fold: int
    """Fold whose checkpoint excluded `tabstar_key` from pretraining — the leakage-free choice."""
    wrong_fold: int
    """A different fold whose checkpoint still included `tabstar_key` — the sanity control."""


# Fold assignments confirmed against `tabstar.tabstar_datasets.PRETRAIN2FOLD` (PR #23 aliases).
# `wrong_fold = (correct_fold + 2) % 5` — deterministic, never collides with `correct_fold`.
OVERLAP_DATASETS = [
    OverlapDataset(
        tabarena_name="jm1",
        tabstar_key="BIN_COMPUTERS_JM1_CODE_DEFECTIONS",
        correct_fold=0,
        wrong_fold=2,
    ),
    OverlapDataset(
        tabarena_name="hiva_agnostic",
        tabstar_key="MUL_SCIENCE_HIV_QSAR",
        correct_fold=1,
        wrong_fold=3,
    ),
    OverlapDataset(
        tabarena_name="wine_quality",
        tabstar_key="REG_FOOD_WINE_QUALITY",
        correct_fold=1,
        wrong_fold=3,
    ),
    OverlapDataset(
        tabarena_name="Diabetes130US",
        tabstar_key="BIN_HEALTHCARE_DIABETES_US130",
        correct_fold=3,
        wrong_fold=0,
    ),
]


class TabSTARModel(AbstractModel):
    """TabArena wrapper around TabSTAR, exposing checkpoint selection as a config hyperparameter.

    Defined in `__main__` scope: this script runs with `debug_mode=True` (in-process backend),
    which is required for a model class defined outside an importable module.
    """

    ag_key = "TABSTAR"
    ag_name = "TabSTAR"

    pretrain_param_name = "pretrain_dataset_or_path"
    """Popped from the hyperparameters in `_fit`; forwarded to `TabSTARClassifier`/`Regressor`.

    `None` -> base checkpoint. A `BIN_`/`REG_`/`MUL_`-prefixed key -> that dataset's excluding-fold
    checkpoint. Any other string -> used verbatim as the HF repo id (for the "wrong" checkpoint).
    """

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs) -> None:
        from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

        X = self.preprocess(X, y=y, is_train=True)
        hps = dict(self._get_model_params())
        pretrain_dataset_or_path = hps.pop(self.pretrain_param_name, None)
        device = "cuda" if num_gpus > 0 else "cpu"
        model_cls = TabSTARClassifier if self.problem_type in ("binary", "multiclass") else TabSTARRegressor
        self.model = model_cls(pretrain_dataset_or_path=pretrain_dataset_or_path, device=device, **hps)
        self.model.fit(X, y)

    def _set_default_params(self) -> None:
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int]:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}


def build_checkpoint_generators(overlap: OverlapDataset) -> list[ConfigGenerator]:
    """The three `ConfigGenerator`s (base/correct/wrong) for one overlapping dataset.

    Each generator's `name` overrides `TabSTARModel.ag_name`, so the three runs show up as
    distinguishable methods (`TabSTAR_base_c1_BAG_L1`, etc.) in the TabArena leaderboard.
    """
    wrong_checkpoint = PRETRAIN_FOLD_REPO_TEMPLATE.format(fold=overlap.wrong_fold)
    checkpoints = {
        "TabSTAR_base": None,
        "TabSTAR_correct": overlap.tabstar_key,
        "TabSTAR_wrong": wrong_checkpoint,
    }
    return [
        ConfigGenerator(
            model_cls=TabSTARModel,
            name=name,
            manual_configs=[{TabSTARModel.pretrain_param_name: checkpoint}],
            search_space={},
        )
        for name, checkpoint in checkpoints.items()
    ]


if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "tabarena_leakage_analysis"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    context = TabArenaContext()
    for overlap in OVERLAP_DATASETS:
        print(f"\n=== Running TabSTAR base/correct/wrong on {overlap.tabarena_name} ===")
        experiments = TabArenaV0pt1ExperimentBundle(
            models=[(gen, 0) for gen in build_checkpoint_generators(overlap)],
        ).build_experiments()
        context.build_and_run_jobs(
            experiments,
            expname=results_dir,
            subset="lite",
            build_kwargs={"dataset_names": [overlap.tabarena_name]},
            new_result_prefix="[Leakage] ",
            debug_mode=True,
        )

    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
