# TabSTAR v1.1 pretraining

The code that produced the TabSTAR v1.1 checkpoints (`alana89/TabSTAR-v1.1-cls-20261011` and
`alana89/TabSTAR-v1.1-reg-20261009`). The model itself (architecture, preprocessing, finetuning) is
reused from the `tabstar` package under `tabstar.tabstarv1p1`; this folder adds what only
pretraining needs:

- `do_t4_eval_fingerprints.py`: fingerprints the evaluation tables T4 data must not contain.
- `do_t4_data.py`: builds a deduplicated, decontaminated pretraining pool from TabLib.
- `do_pretrain.py`: pretrains on that pool, scoring a finetunability Elo after every epoch.
- `do_finetunability_check.py`: the per-epoch finetune-and-score subprocess (called by `do_pretrain.py`).

## Setup

Requires Python 3.11+ and a CUDA GPU with bf16 support. After `source init.sh` (see the main
README), also install `tabarena[benchmark]` and `bencheval` from the
[TabArena repo](https://github.com/autogluon/tabarena); pretraining refuses to start without them.

Run every script from the repo root as a module:

```bash
python -m tabstarv1p1_pretrain.do_pretrain --help
```

No keys are stored in the code. Set them as environment variables, or copy
[`.env.example`](.env.example) to `.env` at the repo root (git-ignored) and fill it in:

| Variable | Effect |
|---|---|
| `WANDB_API_KEY`, `WANDB_ENTITY` | Log to WandB; without a key, WandB is disabled |
| `HF_TOKEN` | Hugging Face access for downloading TabLib/T4 chunks and the fingerprinted HF benchmarks |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle access for `do_t4_eval_fingerprints.py` (MulTaBench tables) |
| `GPU` | Run on `cuda:<GPU>` |
| `CPU` | Force CPU (debugging only; pretraining itself needs a GPU) |
| `MONITOR`, `MEMORY` | Extra timing and GPU-memory logging |

## T4/TabLib data

TabLib is a large public corpus of tables scraped from GitHub. `do_t4_data.py` turns raw TabLib
chunks into dataset directories that pretraining can consume directly: it downloads and extracts
TabLib chunks (caching them under `--hf_cache_dir`), filters candidate columns as prediction
targets with a LightGBM-based learnability check, and writes one directory per accepted target
under `--output_dir`:

```
<output_dir>/cls/<name>/
<output_dir>/reg/<name>/
```

Each holds a `properties.json` and a `train/data.h5`, split by task type. `--d_num` controls the
numeric feature dimension. `--max_datasets` caps the total number of targets across both task
types; `--max_per_task` instead caps `cls` and `reg` independently. The two are mutually exclusive.

```bash
python -m tabstarv1p1_pretrain.do_t4_data --output_dir <path> --max_per_task 200000 --eval_fingerprints <fingerprints.json>
```

### Deduplication and decontamination

TabLib is full of copies, so tables are filtered before any target is curated:

- TabLib files are named `<hash>=__<uuid>.parquet`, and files sharing a hash are copies or
  resamples of one source table, so only the first file per hash is processed, across all chunks.
- Each remaining table is fingerprinted by content: every column with enough distinct values gets a
  MinHash sketch of its value set, which survives row subsets, row and column reordering, and
  column renames (see `pretraining/t4_overlap.py`).
- A table is dropped if it copies an evaluation table, meaning its column-name set equals that
  table's, or at least two of its columns match that table's columns by content.
- A table is dropped if at least two of its columns match an earlier kept T4 table by content.
  Column names alone never count here, since unrelated TabLib tables often share generic headers.
- Tables too small or too large to yield any target are never fingerprinted, so they never shadow
  a larger copy found later.

Every dropped table still gets a rejected verdict per column in the curation cache below, with
reason `duplicate_hash`, `eval_copy` or `near_duplicate` and the table it duplicates in
`duplicate_of`. These verdicts are recomputed on every run, so a new fingerprint file needs no new
`--curation_version`. Decisions follow sorted file order, so a resumed run keeps the same tables.

Blocking evaluation copies needs a fingerprint file of TabArena, BeyondArena, STRABLE and
MulTaBench, built once with:

```bash
python -m tabstarv1p1_pretrain.do_t4_eval_fingerprints --out <fingerprints.json>
```

It downloads every TabArena dataset from OpenML, every BeyondArena and STRABLE source table from
their HF repos (about 11 GB), and every MulTaBench `data.csv` from Kaggle. Without
`--eval_fingerprints`, `do_t4_data.py` still deduplicates but blocks no evaluation copies, and
warns loudly. Resuming keeps every dataset an `--output_dir` already holds, so a pool built under
different filtering needs a fresh `--output_dir`.

### Curation cache

The LightGBM accept/reject verdict per candidate column is the expensive step; verbalizing and
saving an accepted column is cheap. The two are cached separately:

```
<curation_root>/t4_curation_cls_<curation_version>/{accepted,rejected}/<name>/curation.json
<curation_root>/t4_curation_reg_<curation_version>/{accepted,rejected}/<name>/curation.json
```

Bump `--curation_version` to rerun the filtering itself; keep it fixed and point a new
`--output_dir` at it to re-materialize the same accepted columns under different preprocessing
(e.g. a new `--d_num`) without refitting LightGBM.

## Pretraining

`--t4_dir` points at an `--output_dir` produced by `do_t4_data.py`. `--task_type cls`/`reg`
(required) trains one model per task type. `--num_datasets` sets how many datasets each epoch
draws. `--pool_mode active` (default) resamples that subset from the full pool every epoch;
`random` samples it once at startup; `first` takes the first `--num_datasets` in sorted order.

After every epoch, the checkpoint is LoRA-finetuned on a capped subsample of TabArena's fold 0
for each dataset of its task type, and its scores are battled against the baselines in
`finetunability/baseline_scores.csv` (scored under the same protocol) into a finetunability Elo.
The released checkpoints are the epochs with the best finetunability Elo.

The v1.1 recipe, on a decontaminated pool of about 200K tables per task type:

```bash
python -m tabstarv1p1_pretrain.do_pretrain --exp cls50k-active --task_type cls --t4_dir <pool> \
    --num_datasets 50000 --pool_mode active --batch_size 512
python -m tabstarv1p1_pretrain.do_pretrain --exp reg50k-active --task_type reg --t4_dir <pool> \
    --num_datasets 50000 --pool_mode active --batch_size 512
```

Both ran 25 epochs; the released checkpoints are epoch 17 (cls) and epoch 16 (reg). Checkpoints
are saved under `../tabstar_pretrain_models/`.
