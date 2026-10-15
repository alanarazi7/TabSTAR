# Plan: shipping TabSTAR v1.1 as a second model in the `tabstar` package

Status: proposal, nothing implemented yet. This file is the agreed shape of the work before any
code moves.

## Goal

The research repo (`TabSTAR-v2`) has produced two new pretrained checkpoints, branded **TabSTAR
v1.1**: one for classification, one for regression. They are a different architecture and a
different training recipe from the model this package ships today, and they beat it. We want users
to be able to `pip install tabstar` and use either generation.

Three constraints shape everything below:

1. **The two generations share no code.** Duplication is accepted, deliberately. The v1 code stays
   byte-identical to what it is today so published paper results stay reproducible, and the v1.1
   code is copied in from the research repo rather than refactored into a shared abstraction. A
   shared layer would make every future v1.1 change a compatibility question about v1.
2. **The public API does not break.** Today's calling convention keeps working, unchanged, with no
   deprecation warnings in the way of normal use.
3. **It lands in stages.** This document is the first commit. The code follows in reviewable
   phases, each one leaving the package installable and the test suite green.

## Naming

Two naming hazards, settled here so the rest of the document can be unambiguous.

**Package version versus model generation.** The package is already at version `1.1.16`. The new
checkpoints are called "v1.1". These two `1.1`s mean nothing to each other and the collision is
pure accident. Throughout the code and the docs:

- "TabSTAR v1" and "TabSTAR v1.1" always mean the *model generation*, never a package release.
- A package release is always written with three components (`1.1.16`, `1.2.0`).
- The restructure itself ships as **`1.2.0`**: no break, new capability, minor bump. Jumping to
  `2.0.0` would promise a break we are explicitly not making, and would add a third `1.1`-adjacent
  number to the confusion.

**Module names.** A model generation's dot is written `p`, since a module name cannot carry one:

| | package | public estimators |
|---|---|---|
| v1 | `tabstar.tabstarv1` | `TabSTARClassifier`, `TabSTARRegressor` |
| v1.1 | `tabstar.tabstarv1p1` | `TabSTARv1p1Classifier`, `TabSTARv1p1Regressor` |

Symmetric, each name states a model generation, and the top level of `tabstar/` stays free for the
things that belong to the package rather than to a generation.

## Target layout

```
src/tabstar/
    __init__.py            version string, plus the import-compatibility shim (below)
    tabstar_model.py       re-exports all four estimators; the stable import path
    resources/imdb.csv     unchanged, still reachable as files("tabstar")/"resources"
    _compat.py             maps the pre-1.2.0 module paths onto tabstarv1
    tabstarv1/             today's src/tabstar, moved verbatim
        tabstar_model.py tabstar_verbalizer.py tabstar_datasets.py constants.py
        arch/ datasets/ preprocessing/ training/
    tabstarv1p1/            copied from the research repo's tabstar2
        tabstar_model.py tabstar_checkpoints.py constants.py
        arch/ finetune/ preprocessing/ training/ utils/
```

`tabstar_datasets.py` (the `TEXT2FOLD` and `PRETRAIN2FOLD` maps and the paper-fold checkpoint
resolver) is v1 paper machinery and moves into `tabstarv1` with the rest. v1.1 gets its own, much
smaller resolver under a name that says what it does, `tabstarv1p1/tabstar_checkpoints.py`.

## The backwards-compatibility contract

Three tiers, stated explicitly so we know what we are promising.

**Tier 1, guaranteed and tested.** Everything the README and the PyPI page show:

```python
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
```

plus every constructor keyword, `fit`, `predict`, `predict_proba`, `score`,
`score_all_metrics`, `classes_`, `save`, `load` and `download_base_model`, with today's behaviour.
`TabSTARClassifier` keeps meaning the v1 model. A user who upgrades and changes nothing notices
nothing. The package test enforces this.

**Tier 2, shimmed and undocumented.** Deep imports into the old module paths:
`tabstar.training.*`, `tabstar.preprocessing.*`, `tabstar.arch.*`, `tabstar.datasets.*`,
`tabstar.tabstar_verbalizer`, `tabstar.constants`, `tabstar.tabstar_datasets`. These were never
documented as public, but downstream code (including AutoGluon's and TabArena's wrappers, this
repo's own `tabstar_extensions/`, and every model anyone has pickled) reaches into them. They keep
resolving, to the identical objects, via `_compat.py`.

**Tier 3, no promise.** Anything else, including everything under `tabstarv1p1` until the first
release that ships it.

### How the shim works, and why it is not 30 stub files

`_compat.py` installs one `importlib.abc.MetaPathFinder` that resolves `tabstar.<old path>` to
`tabstar.tabstarv1.<old path>` for a fixed list of old paths, and registers the result under both
names in `sys.modules`. It is imported from `tabstar/__init__.py`, which costs nothing: the finder
imports no torch, and a submodule is only materialised when something actually asks for it.

This also solves unpickling for free, which the alternatives do not. `joblib.load` on a model saved
by 1.1.x looks up `tabstar.tabstar_verbalizer`, `tabstar.training.trainer` and friends by their
recorded module path; the finder makes those lookups succeed and hands back the same class objects,
so old pickles keep loading.

Two things this does not cover, both acceptable and both to be stated in the release notes:

- A model saved by 1.2.0 records the new `tabstar.tabstarv1.*` paths and will not load on 1.1.x.
  Forward compatibility was never offered.
- `isinstance` checks against a class imported through the old path still pass, since it is the
  same object, but code that compares `cls.__module__` to a hardcoded string will see the new path.

`tabstar.tabstar_model` is a real module, not a shim target: it re-exports v1's two estimators under
their historical names and v1.1's two under the new names, so one import path offers all four.

## What actually differs between the two estimators

The research repo's estimator was written for a benchmark harness, not for end users, and its
surface drifted. The port conforms it to the v1 contract rather than exposing the drift. Each row
below is a decision, not a description.

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
only blockers in the closure, both trivially rewritten. While we are here: CI's package job runs on
3.9, which contradicts `requires-python` and has only passed because nothing imported has needed
3.10. That job moves to 3.10.

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

## `tabstar_paper`

`tabstar_paper/` reaches into `tabstar.preprocessing.*`, `tabstar.training.*`, `tabstar.arch.*`,
`tabstar.datasets.all_datasets`, `tabstar.constants` and `tabstar.tabstar_model` across roughly a
hundred import sites. Two changes, in this order:

1. **Rewrite the imports to the explicit `tabstar.tabstarv1.*` paths.** Mechanical, and worth doing
   even though the shim would keep them working: paper code that reads `tabstarv1` is
   self-documenting about which generation produced a given number, and it keeps the shim honest as
   a compatibility layer for *other people's* code rather than something we depend on ourselves.
   The same rewrite applies to `tabstar_extensions/`, `test/` and the root-level scripts.
2. **Add v1.1 as a benchmarkable model.** The evaluation and leaderboard code selects a model by
   name; v1.1 becomes another entry, not a flag on the existing one. Existing result files, dataset
   lists and the leakage analysis stay pointed at v1 and keep reproducing.

Nothing in `tabstar_paper` changes meaning in phase 1, so the paper results regenerate identically
across the move. That is the acceptance criterion for the phase.

## Packaging, docs and release

- `pyproject.toml`'s package discovery already globs `src/`, so the new subpackages are picked up
  with no change. `package-data` keeps pointing at `tabstar/resources/*.csv`, which does not move,
  so the quickstart's `files("tabstar").joinpath("resources", "imdb.csv")` is unaffected.
- `README.md` and `README-pypi.md` gain a short section on choosing a generation, with the v1
  quickstart left exactly where it is as the default.
- The two model cards from the research repo's `sota_checkpoints/` ship alongside the code, as the
  honest record of what the checkpoints are and what their published scores were produced with.
- Release `1.2.0` to PyPI once phase 3 is green, and only then.

## Phases

Each phase is a reviewable PR that leaves the package installable and CI green.

**Phase 1, the move.** v1 relocates to `tabstar/tabstarv1/`. `_compat.py` lands with it, along with
the re-exporting `tabstar/tabstar_model.py`. In-repo imports are rewritten. No behaviour changes
anywhere.
*Done when*: the full test suite passes unchanged; a model pickled by 1.1.16 loads; `from
tabstar.tabstar_model import TabSTARClassifier` and a deep import such as `from
tabstar.preprocessing.nulls import MISSING_VALUE` both still work.

**Phase 2, the copy.** The pruned v1.1 modules land under `tabstar/tabstarv1p1/`, dependency-clean
and 3.10-clean, with no public estimator yet.
*Done when*: `import tabstar.tabstarv1p1.tabstar_model` succeeds in a fresh environment installed
from `pyproject.toml` alone, on 3.10 and 3.11.

**Phase 3, the estimators.** `TabSTARv1p1Classifier` and `TabSTARv1p1Regressor`, conformed to the API
table above, with checkpoint resolution and prefetch.
*Done when*: both fit and predict end to end on the bundled `imdb.csv`, under a package test
mirroring the v1 one; a round trip through `save`/`load` reproduces predictions.

**Phase 4, the paper.** v1.1 becomes a selectable model in `tabstar_paper`'s evaluation path.
*Done when*: a v1.1 run produces a leaderboard entry next to v1 on the same dataset list.

**Phase 5, release.** Docs, model cards, version bump, PyPI.

## Open questions

1. Should `TabSTARClassifier` ever come to mean v1.1? Not in 1.2.0. If it should in a later major,
   the shim written in phase 1 is the mechanism, and the decision is worth making explicitly rather
   than drifting into.
2. Whether v1.1 is exposed as a classifier and a regressor, mirroring v1 and sklearn, or as one
   estimator that infers the task from the target and picks its checkpoint accordingly. Two classes
   are the proposal.
3. Whether `tabstar_paper` moves to explicit `tabstar.tabstarv1.*` imports in phase 1 or keeps
   riding the compatibility shim.
4. Whether 1.2.0 goes to PyPI as soon as phase 3 is green, or waits for the paper work.
