"""Quantify TabSTAR's potential pretraining leakage on TabArena-Lite.

TabSTAR is pretrained on a large corpus of tabular datasets, five leave-one-fold-out
checkpoints (`TabSTAR-eval-320-version-fold-k{0..4}`), each excluding a different ~1/5 slice
of the corpus. All 51 of TabArena's datasets are known/curated in TabSTAR's shared dataset
registry (`tabstar2.benchmarks.tabarena.TABARENA` in the TabSTAR-v2 repo maps every one of them
to a `tabstar_paper.datasets.all_datasets.OpenMLDatasetID`), but that registry is broader than
this repo's (v1) pretraining corpus: only 32 of the 51 were actually part of it, i.e. exist as
keys in `tabstar.tabstar_datasets.PRETRAIN2FOLD` / `TEXT2FOLD` (verified by exact key match
against the TabSTAR-v2 name list, and cross-checked against OpenML/Kaggle instance+feature
counts wherever a name alone was ambiguous — see OVERLAP_DATASETS below). The other 19 were
never pretrained on under v1 and are excluded here; TabSTAR-v2 separately excludes all 51 from
its own pretraining pool. Scoring the base checkpoint on one of the 32 risks leakage. This
script quantifies that risk by running each overlapping dataset three ways through TabArena-Lite:

  (a) base    — the public base checkpoint (`alana89/TabSTAR`), pretrained on everything.
  (b) correct — the fold checkpoint that EXCLUDED this dataset from pretraining (leakage-free).
  (c) wrong   — a different fold checkpoint that still INCLUDED this dataset (sanity control:
                if (a) beats (c) similarly to how it beats (b), the gap is not about leakage).

If leakage matters, (a) should score above (b), and (c) should track (a) rather than (b).

Requires the `tabarena` package with its `plot` extra (`pip install "tabarena[plot]"`), which is
not a declared dependency of this repo. The bare package is enough for `build_and_run_jobs()`, but
`context.compare()` at the end lazily imports `tueplots`/`matplotlib`/`seaborn`/`autorank`/
`adjusttext` to build the leaderboard and figures, so a bare `pip install tabarena` fails there
only after every dataset has already finished training.

Results land under this file's directory: raw per-run TabArena job artifacts in
`experiments/tabarena_leakage_analysis/`, the compared leaderboard and figures in
`eval/tabarena_leakage_analysis/`.

Usage:
    python tabarena_leakage_analysis.py                    # all overlapping datasets
    python tabarena_leakage_analysis.py --tabarena_name diabetes   # a single dataset

    # Cluster array job: one dataset per worker, no `[plot]` extra needed on workers.
    python tabarena_leakage_analysis.py --tabarena_name <name> --skip_compare
    # Once every worker has finished, aggregate on any machine with `tabarena[plot]` installed:
    python tabarena_leakage_analysis.py --compare_only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator
from tabstar.tabstar_datasets import PRETRAIN2FOLD, TEXT2FOLD

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


def correct_fold(tabstar_key: str) -> int:
    """The fold whose checkpoint excluded `tabstar_key` from pretraining (leakage-free choice).

    Mirrors `tabstar.tabstar_datasets._get_tabstar_version_from_dataset`'s lookup order:
    TEXT2FOLD first, then PRETRAIN2FOLD.
    """
    fold = TEXT2FOLD.get(tabstar_key, PRETRAIN2FOLD.get(tabstar_key))
    if fold is None:
        raise ValueError(f"{tabstar_key} not found in TEXT2FOLD or PRETRAIN2FOLD")
    return fold


def wrong_fold(tabstar_key: str) -> int:
    """A different fold whose checkpoint still included `tabstar_key` (sanity control)."""
    return (correct_fold(tabstar_key) + 2) % 5


# Verified against `tabstar.tabstar_datasets.PRETRAIN2FOLD` / `TEXT2FOLD`: exact name match to
# the dataset's real OpenML name, or (where names diverged) an exact match on OpenML/Kaggle
# instance+feature counts against TabArena's `curated_tabarena_dataset_metadata.csv` row. The
# other 19 of TabArena's 51 datasets were checked the same way and have no corpus overlap.
OVERLAP_DATASETS = [
    OverlapDataset(tabarena_name="jm1", tabstar_key="BIN_COMPUTERS_JM1_CODE_DEFECTIONS"),
    OverlapDataset(tabarena_name="hiva_agnostic", tabstar_key="MUL_SCIENCE_HIV_QSAR"),
    OverlapDataset(tabarena_name="wine_quality", tabstar_key="REG_FOOD_WINE_QUALITY"),
    OverlapDataset(tabarena_name="Diabetes130US", tabstar_key="BIN_HEALTHCARE_DIABETES_US130"),
    OverlapDataset(tabarena_name="airfoil_self_noise", tabstar_key="REG_SCIENCE_AIRFOIL_SELF_NOISE"),
    OverlapDataset(tabarena_name="Amazon_employee_access", tabstar_key="BIN_PROFESSIONAL_AMAZON_EMPLOYEE_ACCESS"),
    OverlapDataset(tabarena_name="anneal", tabstar_key="MUL_SCIENCE_ANNEAL_CHEMICAL"),
    OverlapDataset(tabarena_name="APSFailure", tabstar_key="BIN_ANONYM_APS_FAILURE"),
    OverlapDataset(tabarena_name="bank-marketing", tabstar_key="BIN_FINANCIAL_BANK_MARKETING"),
    OverlapDataset(tabarena_name="Bioresponse", tabstar_key="BIN_ANONYM_BIORESPONSE"),
    OverlapDataset(tabarena_name="blood-transfusion-service-center", tabstar_key="BIN_HEALTHCARE_BLOOD_TRANSFUSION"),
    OverlapDataset(tabarena_name="churn", tabstar_key="BIN_CONSUMER_CHURN_TELEPHONY"),
    OverlapDataset(tabarena_name="concrete_compressive_strength", tabstar_key="REG_SCIENCE_CONCRETE_COMPRESSIVE_STRENGTH"),
    OverlapDataset(tabarena_name="credit-g", tabstar_key="BIN_FINANCIAL_CREDIT_GERMAN"),
    OverlapDataset(tabarena_name="diamonds", tabstar_key="REG_CONSUMER_DIAMONDS_PRICES"),
    OverlapDataset(tabarena_name="GiveMeSomeCredit", tabstar_key="BIN_FINANCIAL_CREDIT_GIVE_ME_SOME"),
    OverlapDataset(tabarena_name="kddcup09_appetency", tabstar_key="BIN_ANONYM_KDDCUP_09_APPETENCY"),
    OverlapDataset(tabarena_name="miami_housing", tabstar_key="REG_HOUSES_MIAMI"),
    OverlapDataset(tabarena_name="online_shoppers_intention", tabstar_key="BIN_CONSUMER_ONLINE_SHOPPERS_PURCHASE_INTENTION"),
    OverlapDataset(tabarena_name="physiochemical_protein", tabstar_key="REG_SCIENCE_PHYSIOCHEMICAL_PROTEIN"),
    OverlapDataset(tabarena_name="qsar-biodeg", tabstar_key="BIN_SCIENCE_QSAR_BIODEG"),
    OverlapDataset(tabarena_name="QSAR-TID-11", tabstar_key="REG_SCIENCE_QSAR_TID_11"),
    OverlapDataset(tabarena_name="QSAR_fish_toxicity", tabstar_key="REG_NATURE_FISH_TOXICITY"),
    OverlapDataset(tabarena_name="splice", tabstar_key="MUL_GENETICS_SPLICE_DNA"),
    OverlapDataset(tabarena_name="superconductivity", tabstar_key="REG_SCIENCE_SUPERCONDUCTIVITY"),
    OverlapDataset(tabarena_name="website_phishing", tabstar_key="MUL_COMPUTERS_PHISHING_WEBSITE_HUDDERSFIELD"),
    OverlapDataset(tabarena_name="heloc", tabstar_key="BIN_FINANCIAL_CREDIT_FICO_HELOC"),
    OverlapDataset(tabarena_name="Bank_Customer_Churn", tabstar_key="BIN_FINANCIAL_BANK_CUSTOMER_CHURN_SHRUTIME"),
    OverlapDataset(tabarena_name="credit_card_clients_default", tabstar_key="BIN_FINANCIAL_CC_TAIWAN_CREDIT_DEFAULT"),
    OverlapDataset(tabarena_name="diabetes", tabstar_key="BIN_HEALTHCARE_DIABETES_RISK_FACTORS"),
    OverlapDataset(tabarena_name="healthcare_insurance_expenses", tabstar_key="REG_FINANCIAL_INSURANCE_PREMIUM_DATA"),
    OverlapDataset(tabarena_name="houses", tabstar_key="REG_HOUSES_CALIFORNIA_HOUSES"),
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

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        **kwargs,
    ) -> None:
        from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

        X = self.preprocess(X, y=y, is_train=True)
        hps = dict(self._get_model_params())
        pretrain_dataset_or_path = hps.pop(self.pretrain_param_name, None)
        device = "cuda" if num_gpus > 0 else "cpu"
        model_cls = TabSTARClassifier if self.problem_type in ("binary", "multiclass") else TabSTARRegressor
        # Forward AutoGluon's per-fold budget (TabArena's own TabSTAR wrapper does the same). Without
        # it a fit never self-limits, and AutoGluon aborts the whole 8-fold bag with TimeLimitExceeded
        # as soon as the folds so far project past the bag's 1 h budget (after fold 1: any fit > 450 s).
        self.model = model_cls(
            pretrain_dataset_or_path=pretrain_dataset_or_path, device=device, time_limit=time_limit, **hps
        )
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


class TabSTARBaseModel(TabSTARModel):
    """The base checkpoint variant, pretrained on everything."""

    ag_key = "TabSTAR_base"
    ag_name = "TabSTAR_base"


class TabSTARCorrectModel(TabSTARModel):
    """The fold checkpoint that EXCLUDED this dataset from pretraining (leakage-free)."""

    ag_key = "TabSTAR_correct"
    ag_name = "TabSTAR_correct"


class TabSTARWrongModel(TabSTARModel):
    """A different fold checkpoint that still INCLUDED this dataset (sanity control)."""

    ag_key = "TabSTAR_wrong"
    ag_name = "TabSTAR_wrong"


def build_checkpoint_generators(overlap: OverlapDataset) -> list[ConfigGenerator]:
    """The three `ConfigGenerator`s (base/correct/wrong) for one overlapping dataset.

    TabArena's bagged-experiment naming keys off `model_cls.ag_name`/`ag_key` (not
    `ConfigGenerator(name=...)`, which the naming path ignores), so each variant is its own named
    `TabSTARModel` subclass — a `type()`-generated class isn't picklable (AutoGluon persists the
    predictor to disk mid-fit), so these must be real module-level classes, not built dynamically
    per dataset. The checkpoint itself still varies per dataset, passed as a hyperparameter.
    """
    variants = {
        TabSTARBaseModel: None,
        TabSTARCorrectModel: overlap.tabstar_key,
        TabSTARWrongModel: PRETRAIN_FOLD_REPO_TEMPLATE.format(fold=wrong_fold(overlap.tabstar_key)),
    }
    return [
        ConfigGenerator(
            model_cls=model_cls,
            manual_configs=[{TabSTARModel.pretrain_param_name: checkpoint}],
            search_space={},
        )
        for model_cls, checkpoint in variants.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabarena_name",
        default=None,
        help="Restrict to a single OVERLAP_DATASETS entry by its tabarena_name (default: run all).",
    )
    parser.add_argument(
        "--skip_compare",
        action="store_true",
        help="Skip context.compare() after training (for per-dataset cluster jobs; requires "
        "only the bare tabarena package, not its [plot] extra). Aggregate later with --compare_only.",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Skip training and only run context.compare() over already-persisted results "
        "(requires `pip install \"tabarena[plot]\"`). Mutually exclusive with --tabarena_name.",
    )
    args = parser.parse_args()
    if args.compare_only and args.tabarena_name is not None:
        parser.error("--compare_only runs over all overlapping datasets; drop --tabarena_name.")
    if args.compare_only and args.skip_compare:
        parser.error("--compare_only and --skip_compare are mutually exclusive.")
    return args


if __name__ == "__main__":
    args = parse_args()
    datasets = OVERLAP_DATASETS
    if args.tabarena_name is not None:
        datasets = [d for d in OVERLAP_DATASETS if d.tabarena_name == args.tabarena_name]
        if not datasets:
            raise ValueError(f"{args.tabarena_name!r} not found in OVERLAP_DATASETS")

    here = Path(__file__).parent
    run_name = "tabarena_leakage_analysis"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    context = TabArenaContext()
    if not args.compare_only:
        for overlap in datasets:
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

    if args.skip_compare:
        print(f"\nSkipping compare(); results persisted under {results_dir}")
    else:
        leaderboard = context.compare(output_dir=eval_dir)
        leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
        print("\n=== TabArena leaderboard (website format) ===")
        print(leaderboard_website.to_markdown(index=False))
        print(f"\nView saved figures in {eval_dir}")
