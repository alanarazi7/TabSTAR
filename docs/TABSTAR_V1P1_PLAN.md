# Plan: shipping TabSTAR v1.1 as a second model in the `tabstar` package

Status: proposal, nothing implemented yet.

## Goal

The research repo (`TabSTAR-v2`) has produced two new pretrained checkpoints, branded **TabSTAR
v1.1**: one for classification, one for regression. They are a different architecture and a
different training recipe from the model this package ships today, and they beat it. We want users
to be able to `pip install tabstar` and use either generation.

Three constraints shape everything below:

1. **Nothing that exists today is edited.** This work only adds files. No module moves, nothing is
   renamed, no import path changes meaning. Backwards compatibility is therefore not a mechanism we
   have to build and test; it is a property of the diff.
2. **The two generations share no code.** Duplication is accepted, deliberately. The v1.1 code is
   copied in from the research repo rather than refactored into anything shared with v1. A shared
   layer would make every future v1.1 change a compatibility question about v1, and would mean
   touching v1, which constraint 1 forbids.
3. **It lands in stages.** This document is the first commit. The code follows in reviewable
   phases, each one leaving the package installable and the test suite green.

## Naming

Two naming hazards, settled here so the rest of the document can be unambiguous.

**Package version versus model generation.** The package is already at version `1.1.16`. The new
checkpoints are called "v1.1". These two `1.1`s mean nothing to each other and the collision is
pure accident. Throughout the code and the docs:

- "TabSTAR v1" and "TabSTAR v1.1" always mean the *model generation*, never a package release.
- A package release is always written with three components (`1.1.16`, `1.2.0`).
- This ships as **`1.2.0`**: no break, new capability, minor bump.

**Module names.** A model generation's dot is written `p`, since a module name cannot carry one:

| | implementation package | import path | public estimators |
|---|---|---|---|
| v1 | `tabstar` itself, as today | `tabstar.tabstar_model` | `TabSTARClassifier`, `TabSTARRegressor` |
| v1.1 | `tabstar.tabstarv1p1` | `tabstar.tabstar_v1p1_model` | `TabSTARv1p1Classifier`, `TabSTARv1p1Regressor` |

Each generation has its own entry module at the top level, and the historical one does not change
meaning. A user asks for v1.1 by importing a different module, never by getting different behaviour
out of the same one. `TabSTARClassifier` keeps meaning the v1 model, in this release and in future
ones.

## Target layout

Everything already in `src/tabstar/` stays exactly where it is and as it is. Two new entries
appear next to it:

```
src/tabstar/
    __init__.py              unchanged
    tabstar_model.py         unchanged, still v1's estimators
    tabstar_verbalizer.py    unchanged
    tabstar_datasets.py      unchanged
    constants.py             unchanged
    arch/ datasets/ preprocessing/ training/   unchanged
    resources/imdb.csv       unchanged

    tabstar_v1p1_model.py    NEW: v1.1's estimators
    tabstarv1p1/             NEW: copied from the research repo's tabstar2
        tabstar_model.py tabstar_checkpoints.py constants.py
        arch/ finetune/ preprocessing/ training/ utils/
```

`tabstar_v1p1_model.py` is thin: it re-exports v1.1's two estimators and nothing else. The work
lives in `tabstarv1p1/`.

Nothing inside `tabstarv1p1/` imports anything outside it, apart from third-party packages. The
research repo's `tabstar2` already has no import of the public `tabstar` package, so this holds on
arrival and is worth keeping as a rule: the two generations are independent subtrees that happen to
be shipped in one wheel.

## What actually differs between the two estimators

The research repo's estimator was written for a benchmark harness, not for end users, and its
surface drifted. The port conforms it to the v1 contract rather than exposing the drift, so that a
user who knows one estimator can drive the other. Each row below is a decision, not a description.

| | v1 (today, public) | v1.1 (research repo) | what the port does |
|---|---|---|---|
| `fit` signature | `fit(X, y, x_val=None, y_val=None)` | `fit(X, y)`, splits its own validation | accept `x_val`/`y_val` |
| `fit` return | `self` | best validation loss | return `self` (sklearn convention); the loss stays reachable as a fitted attribute |
| classifier `predict` | original labels, inverse-transformed | integer class indices | original labels |
| `score` | `score(X, y)` | `score(X, y, is_cls)` | `score(X, y)`; the task is already known from the class |
| `save` / `load` | joblib round trip | absent | ported from v1 |
| `download_base_model()` | static, prefetches the checkpoint | absent | ported, resolving both checkpoints |
| `classes_` | property | absent | ported from v1 |
| checkpoint selection | one default repo | two, keyed on task | see below |
| `time_limit` | honoured, AutoGluon passes it | absent | **must** be ported, or v1.1 cannot be benchmarked under AutoGluon |
| `keep_model`, `output_dir`, `val_batch_size` | present | absent | port, same meaning |
| `lora_alpha` | constructor argument | hardcoded to `2 * r` | keep hardcoded; expose only if a reason appears |
| `is_paper_version` | selects the paper checkpoint and disables checkpoint averaging | no equivalent concept | not ported; v1.1 has no paper variant |
| `global_batch` | that name | `lora_global_batch` | use v1's name |

"Ported from v1" means copied, not imported.

Two v1.1 behaviours that have no v1 counterpart and are kept as they are:

- **Reusing a preloaded base model.** The research harness finetunes many datasets from one
  in-memory base model. The constructor keeps accepting one, and keeps unloading the LoRA adapter
  if a fit raises, so a crash cannot corrupt a base another fit is about to use.
- **Checkpoint averaging is gone.** The research repo has deleted it (TabSTAR-v2 PR #148); finetuning
  keeps the single best epoch's weights, and the LoRA schedule anneals once per optimizer update
  rather than once per epoch. The port carries the post-deletion code. Note that this means the
  published v1.1 checkpoint scores, which were produced with averaging on, are not reproducible
  from the shipped code; the model cards say so.

## What gets copied, and what gets pruned

The import closure of the research repo's estimator is 57 modules and roughly 3,200 lines. It is not
all shippable. Three things have to happen on the way in:

**Prune the research tail.** The closure reaches pretraining data machinery (HDF5 shard readers,
pretrain argument parsing, run monitoring, W&B logging) through a handful of single-constant
imports: the model config wants one text batch size and one unfreeze-layer count, and the LoRA
wrapper wants one model-summary helper. Re-home those few values into the v1.1 package's own
constants module and the rest of that subtree never comes along.

**Add no dependency.** The package today depends on joblib, numpy, pandas, peft, scikit-learn,
skrub, torch, tqdm and transformers. The copied code must not widen that. W&B, python-dotenv,
psutil and gpustat appear in the research modules (logging, device reporting, hardware stats) and
all of it is stripped rather than made optional.

**Make it import on the package's floor Python.** The research repo pins `==3.11.*`; the package
declares `>=3.10`. One `typing.Self` and one PEP 604 union in a runtime-evaluated annotation are the
only blockers in the closure, both trivially rewritten.

CI's package job runs on 3.9, which already contradicts `requires-python >=3.10` and has only
passed because nothing imported has needed 3.10. Nothing breaks while no test imports v1.1, so this
is not urgent; the phase that adds a v1.1 test moves that job to 3.10, and it is the one existing
file that phase touches.

## Checkpoints

v1 resolves a single default repo (`alana89/TabSTAR`). v1.1 has one per task:

- `alana89/TabSTAR-v1.1-cls-20261011`
- `alana89/TabSTAR-v1.1-reg-20261009`

so the resolver takes the task type, and `TabSTARv1p1Classifier` and `TabSTARv1p1Regressor` each know
their own. `pretrain_dataset_or_path` keeps working as the override, same as v1.

The research repo loads these straight through `from_pretrained` with no explicit prefetch, which is
fine on a developer box and wrong for a package: the v1 estimator deliberately calls an explicit
download step before fitting, so a missing or unreachable checkpoint fails early with a clear
message instead of mid-fit. The port adopts v1's prefetch, for both the checkpoint and the E5
tokenizer, and keeps the static `download_base_model()` entry point for users who want to warm the
cache ahead of time.

## The research half of the repo stays untouched

`tabstar_paper/` (pretraining, dataset curation, the baselines and the leaderboard),
`tabstar_extensions/`, `test/` and the root-level scripts import from `tabstar.*` across roughly a
hundred sites. None of them are edited, and none of them need to be: every path they import still
resolves to the same object, because nothing moved. Paper results regenerate identically, for the
trivial reason that no code they run has changed.

Benchmarking v1.1 through the paper harness is separate, additive work, not part of this. Until it
happens, the paper code means v1 everywhere, which is what it already means.

## Packaging, docs and release

- `pyproject.toml`'s package discovery already globs `src/`, so `tabstarv1p1` is picked up with no
  change to the build configuration. The version bump is the only edit to that file.
- `README.md` and `README-pypi.md` gain a short section on choosing a generation, with the v1
  quickstart left exactly where it is as the default.
- The two model cards from the research repo's `sota_checkpoints/` ship alongside the code, as the
  honest record of what the checkpoints are and what their published scores were produced with.
- Release `1.2.0` to PyPI once phase 2 is green, and only then.

## Phases

Each phase is a reviewable PR that leaves the package installable and CI green.

**Phase 1, the copy.** The pruned v1.1 modules land under `src/tabstar/tabstarv1p1/`,
dependency-clean and 3.10-clean, with no public estimator yet. Purely new files.
*Done when*: `import tabstar.tabstarv1p1.tabstar_model` succeeds in a fresh environment installed
from `pyproject.toml` alone, on 3.10 and 3.11; `git diff --stat` shows additions only.

**Phase 2, the estimators.** `src/tabstar/tabstar_v1p1_model.py`, exposing `TabSTARv1p1Classifier`
and `TabSTARv1p1Regressor` conformed to the API table above, with checkpoint resolution and
prefetch. Adds a package test mirroring the v1 one, and moves the CI package job to 3.10 so that
test can run.
*Done when*: both fit and predict end to end on the bundled `imdb.csv`; a round trip through
`save`/`load` reproduces predictions; the v1 tests still pass untouched.

**Phase 3, release.** Docs, model cards, version bump, `1.2.0` to PyPI.

## Deferred: splitting v1 into its own subpackage

An earlier draft moved today's `src/tabstar` into `src/tabstar/tabstarv1/` for symmetry, with an
import shim keeping the old paths alive for downstream code and for unpickling old models. That is
real work with real risk and it buys tidiness, not capability. It is not part of this plan. If the
symmetry is ever worth it, it is a standalone change made on its own merits, after v1.1 has shipped.
