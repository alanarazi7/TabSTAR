import os

# Must be set before numpy/lightgbm are imported: their thread pools are sized once at
# library-load time and that sizing survives forking worker processes, so setting these
# later (e.g. inside create_t4_data) has no effect. With `workers` processes running
# concurrently, unbounded per-process threading causes massive CPU oversubscription.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

# LightGBM ignores OMP_NUM_THREADS and auto-detects hardware core count instead; the
# LightGBM(num_threads=...) calls below cap it explicitly per fit.
LGBM_NUM_THREADS = 1

import glob
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from os.path import join, exists
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from lightgbm.basic import LightGBMError
from pandas import DataFrame, Series
from tqdm import tqdm

from tabstar.tabstarv1p1.preprocessing.splits import split_to_test

from tabstarv1p1_pretrain.baselines.lgbm import LightGBM
from tabstarv1p1_pretrain.pretraining.hdf5 import RealDataset, RealDatasetProperties, save_pretrain_dataset
from tabstarv1p1_pretrain.pretraining.t4_overlap import OverlapIndex, frame_signatures, load_fingerprint_index
from tabstar.tabstarv1p1.preprocessing.tabstar_verbalizer import TabSTARv1p1Verbalizer
from tabstarv1p1_pretrain.datasets.objects import SupervisedTask
from tabstarv1p1_pretrain.utils.io_handlers import dump_json, load_json, unzip_file
from tabstarv1p1_pretrain.utils.warnings import silence_baselines_prints

T4_REPO_ID = "mlfoundations/t4-full"
DEFAULT_HF_CACHE_DIR = "../t4_hf_cache_dir"
DEFAULT_OUTPUT_DIR = "../t4_pretrain_data"

MIN_PRETRAIN_ROWS = 500
MAX_PRETRAIN_ROWS = 100_000
MIN_PRETRAIN_FEATURES = 5
MAX_PRETRAIN_FEATURES = 150
MIN_REGRESSION_UNIQUE = 50
MAX_CLS_UNIQUE = 20
MIN_CLS_EXAMPLES = 10
MAX_AUC = 0.98
MAX_RSQ = 0.95
MIN_AUC = 0.55
MIN_RSQ = 0.7

CURATION_FILE = "curation.json"

REASON_TOO_FEW_ROWS = "too_few_rows"
REASON_TOO_FEW_FEATURES = "too_few_features"
REASON_TOO_MANY_FEATURES = "too_many_features"
REASON_TOO_MANY_CLASSES = "too_many_classes"
REASON_RARE_CLASS = "rare_class"
REASON_TOO_FEW_UNIQUE_REGRESSION = "too_few_unique_regression"
REASON_UNSEEN_TEST_CLASS = "unseen_test_class"
REASON_LIGHTGBM_ERROR = "lightgbm_error"
REASON_SCORE_TOO_LOW = "score_too_low"
REASON_SCORE_TOO_HIGH = "score_too_high"
REASON_NULL_TARGET = "null_target"
REASON_DUPLICATE_HASH = "duplicate_hash"
REASON_EVAL_COPY = "eval_copy"
REASON_NEAR_DUPLICATE = "near_duplicate"
# Decided per table against the rest of the corpus, so a cached one is never reused for a kept table.
TABLE_REJECT_REASONS = (REASON_DUPLICATE_HASH, REASON_EVAL_COPY, REASON_NEAR_DUPLICATE)


@dataclass
class CurationOutcome:
    """The accept/reject verdict for one candidate (parquet_file, column) target, cached so a
    later preprocessing change can skip re-running the LightGBM-based filtering."""
    name: str
    parquet_file: str
    column: str
    is_cls: bool
    kept: bool
    reject_reason: Optional[str]
    score: Optional[float]
    train_size: Optional[int]
    duplicate_of: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "CurationOutcome":
        return cls(**load_json(path))

    def save(self, path: str) -> None:
        dump_json(asdict(self), path=path)


def list_t4_chunks() -> List[str]:
    """Chunk zip filenames available in the T4 repo, sorted so chunks are consumed in order."""
    files = list_repo_files(repo_id=T4_REPO_ID, repo_type="dataset")
    return sorted(f for f in files if re.fullmatch(r"chunk-\d+\.zip", f))


def ensure_chunk_ready(chunk_filename: str, hf_cache_dir: str) -> str:
    """Downloads and extracts one chunk if it isn't already on disk, returns its extracted dir."""
    extracted_dir = join(hf_cache_dir, chunk_filename[:-len(".zip")])
    if exists(extracted_dir):
        return extracted_dir
    zip_path = join(hf_cache_dir, chunk_filename)
    if not exists(zip_path):
        print(f"⬇️  Downloading {chunk_filename}")
        hf_hub_download(repo_id=T4_REPO_ID, repo_type="dataset", filename=chunk_filename, local_dir=hf_cache_dir)
    print(f"📦 Extracting {zip_path}")
    unzip_file(zip_path)
    return extracted_dir


def create_t4_data(hf_cache_dir: str, output_dir: str, d_num: int, workers: int, curation_root: str,
                    curation_version: str, eval_fingerprints: Optional[str], max_datasets: Optional[int] = None,
                    max_per_task: Optional[int] = None) -> List[str]:
    if (max_datasets is None) == (max_per_task is None):
        raise ValueError("Pass exactly one of max_datasets or max_per_task")
    if eval_fingerprints is None:
        print("🚨🚨🚨 No --eval_fingerprints given: T4 copies of evaluation tables will NOT be blocked")
        eval_index = OverlapIndex()
    else:
        eval_index = load_fingerprint_index(eval_fingerprints)
        print(f"🛡️  Blocking T4 copies of {eval_index.size} fingerprinted evaluation columns from {eval_fingerprints}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hf_cache_dir, exist_ok=True)
    # Cap per-process thread pools before forking workers, so LightGBM/BLAS don't each
    # spawn one thread per core: with `workers` processes running concurrently, unbounded
    # per-process threading causes massive CPU oversubscription.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    cls_prefix = join(output_dir, "cls") + os.sep
    saved_dirs: List[str] = _existing_materialized_dirs(output_dir)
    n_cls = sum(1 for d in saved_dirs if d.startswith(cls_prefix))
    n_reg = len(saved_dirs) - n_cls
    if saved_dirs:
        print(f"↩️  Resuming: {len(saved_dirs)} datasets already materialized in {output_dir} "
              f"({n_cls} cls, {n_reg} reg)")
    target_desc = f"{max_per_task} per task" if max_per_task is not None else f"{max_datasets} total"
    if _reached_target(n_cls, n_reg, max_datasets, max_per_task):
        return _trim(saved_dirs, cls_prefix, max_datasets, max_per_task)
    chunks = list_t4_chunks()
    print(f"T4 corpus has {len(chunks)} chunks; pulling only as many as needed for {target_desc}")
    first_by_hash: Dict[str, str] = {}
    t4_index = OverlapIndex()
    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        prefetch_future = None
        for i, chunk_filename in enumerate(chunks):
            if _reached_target(n_cls, n_reg, max_datasets, max_per_task):
                break
            extracted_dir = prefetch_future.result() if prefetch_future else ensure_chunk_ready(chunk_filename,
                                                                                                 hf_cache_dir)
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
            prefetch_future = (prefetch_pool.submit(ensure_chunk_ready, next_chunk, hf_cache_dir)
                                if next_chunk is not None else None)
            all_parquet_files = sorted(glob.glob(join(extracted_dir, "**", "*.parquet"), recursive=True))
            unique_hash_files, hash_duplicates = first_member_per_hash(all_parquet_files, first_by_hash)
            rejections = {f: (REASON_DUPLICATE_HASH, _pretty_parquet_name(first, hf_cache_dir))
                          for f, first in hash_duplicates.items()}
            rejections.update(_screen_overlapping_tables(unique_hash_files, eval_index, t4_index, workers,
                                                         chunk_filename, hf_cache_dir))
            print(f"{chunk_filename}: {len(all_parquet_files) - len(rejections)} tables go to curation out of "
                  f"{len(all_parquet_files)} parquet files ({len(unique_hash_files)} unique hashes)")
            # Snapshotted once per chunk, not live per-file: skipping a task's already-satisfied
            # columns saves the expensive LightGBM fit on them for the rest of this chunk.
            need_cls = max_per_task is None or n_cls < max_per_task
            need_reg = max_per_task is None or n_reg < max_per_task
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(create_datasets_for_parquet_file, f, d_num, output_dir, hf_cache_dir,
                                       curation_root, curation_version, need_cls, need_reg, rejections.get(f)): f
                           for f in all_parquet_files}
                desc = f"Preparing T4 datasets ({chunk_filename})"
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    f = futures[future]
                    try:
                        dirs = future.result()
                    except Exception as e:
                        print(f"⚠️ Failed on {f}: {type(e).__name__}: {e}")
                        continue
                    for d in dirs:
                        saved_dirs.append(d)
                        if d.startswith(cls_prefix):
                            n_cls += 1
                        else:
                            n_reg += 1
                    if dirs:
                        print(f"💾 {len(saved_dirs)} datasets saved so far ({n_cls} cls, {n_reg} reg), "
                              f"target {target_desc}")
                    if _reached_target(n_cls, n_reg, max_datasets, max_per_task):
                        for pending in futures:
                            pending.cancel()
                        break
    return _trim(saved_dirs, cls_prefix, max_datasets, max_per_task)


def first_member_per_hash(parquet_files: List[str],
                          first_by_hash: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    """T4 members are named `<hash>=__<uuid>.parquet`, and members sharing a hash are copies or
    resamples of one source table, so only the first member per hash is kept. Returns the kept
    files and each duplicate's first member. `first_by_hash` is updated in place so dedup also
    holds across chunks."""
    kept, duplicates = [], {}
    for f in parquet_files:
        table_hash = os.path.basename(f).split("=__")[0]
        if table_hash in first_by_hash:
            duplicates[f] = first_by_hash[table_hash]
            continue
        first_by_hash[table_hash] = f
        kept.append(f)
    return kept, duplicates


def _screen_overlapping_tables(parquet_files: List[str], eval_index: OverlapIndex, t4_index: OverlapIndex,
                               workers: int, chunk_filename: str, hf_cache_dir: str) -> Dict[str, Tuple[str, str]]:
    """(reject_reason, duplicate_of) for every table copying an evaluation table or near-duplicating
    a table kept earlier. Tables too small or large to yield a target are neither rejected here nor
    indexed. Decisions run in sorted file order, so a resumed run keeps the same tables."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        desc = f"Fingerprinting T4 tables ({chunk_filename})"
        signatures = list(tqdm(pool.map(_table_signatures, parquet_files, chunksize=16), total=len(parquet_files),
                               desc=desc))
    rejections = {}
    n_out_of_range = 0
    for f, signature in zip(parquet_files, signatures):
        if signature is None:
            n_out_of_range += 1
            continue
        schema, sketches = signature
        eval_match = eval_index.blocking_table(schema, sketches)
        if eval_match is not None:
            eval_table, evidence = eval_match
            print(f"🛡️  Blocked {f}: copies {eval_table} ({evidence})")
            rejections[f] = (REASON_EVAL_COPY, eval_table)
            continue
        # Content only: unrelated T4 tables often share a generic header.
        pq_name = _pretty_parquet_name(f, hf_cache_dir)
        t4_match = t4_index.blocking_table(None, sketches)
        if t4_match is not None:
            rejections[f] = (REASON_NEAR_DUPLICATE, t4_match[0])
            continue
        t4_index.add_table(pq_name, None, sketches)
    n_eval_copies = sum(reason == REASON_EVAL_COPY for reason, _ in rejections.values())
    print(f"{chunk_filename}: {n_out_of_range} out-of-range tables, {n_eval_copies} evaluation copies, "
          f"{len(rejections) - n_eval_copies} near-duplicates")
    return rejections


def _table_signatures(f: str) -> Optional[Tuple[str, List[List[int]]]]:
    """None for a table whose size rules out every target, so it never enters the dedup index."""
    try:
        pq_df = _read_t4_table(f)
    except Exception as e:
        print(f"⚠️ Failed to read {f}: {type(e).__name__}: {e}")
        return None
    n_features = len(pq_df.columns) - 1
    if len(pq_df) < MIN_PRETRAIN_ROWS:
        return None
    assert len(pq_df) <= MAX_PRETRAIN_ROWS, f"{f}: {len(pq_df)=} exceeds {MAX_PRETRAIN_ROWS=}"
    if not MIN_PRETRAIN_FEATURES <= n_features <= MAX_PRETRAIN_FEATURES:
        return None
    return frame_signatures(pq_df)


def _read_t4_table(f: str) -> DataFrame:
    # use_threads=False: pyarrow's own thread pool ignores OMP_NUM_THREADS and otherwise
    # spins up one thread per core per worker process, on top of process-level parallelism.
    pq_df = pd.read_parquet(f, use_threads=False)
    pq_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    return pq_df


def _reached_target(n_cls: int, n_reg: int, max_datasets: Optional[int], max_per_task: Optional[int]) -> bool:
    if max_per_task is not None:
        return n_cls >= max_per_task and n_reg >= max_per_task
    return (n_cls + n_reg) >= max_datasets


def _trim(saved_dirs: List[str], cls_prefix: str, max_datasets: Optional[int],
         max_per_task: Optional[int]) -> List[str]:
    if max_per_task is None:
        kept, excess = saved_dirs[:max_datasets], saved_dirs[max_datasets:]
    else:
        cls_dirs = [d for d in saved_dirs if d.startswith(cls_prefix)]
        reg_dirs = [d for d in saved_dirs if not d.startswith(cls_prefix)]
        kept = cls_dirs[:max_per_task] + reg_dirs[:max_per_task]
        excess = cls_dirs[max_per_task:] + reg_dirs[max_per_task:]
    # Materialization writes to disk immediately per column; a worker racing past the target
    # before the stop check propagates leaves extra directories that were never in `kept`.
    for d in excess:
        shutil.rmtree(d, ignore_errors=True)
    return kept


def create_datasets_for_parquet_file(f: str, d_num: int, output_dir: str, hf_cache_dir: str, curation_root: str,
                                     curation_version: str, need_cls: bool, need_reg: bool,
                                     table_rejection: Optional[Tuple[str, str]] = None) -> List[str]:
    """Treats every column of the parquet file as a candidate target, keeping every one that
    survives the learnability filter as its own pretrain dataset. A `table_rejection` of
    (reject_reason, duplicate_of) rejects every column instead. Module-level so it can be
    submitted to a process pool.
    """
    pq_df = _read_t4_table(f)
    pq_name = _pretty_parquet_name(f, hf_cache_dir=hf_cache_dir)
    if table_rejection is not None:
        _reject_table(pq_df, pq_name, curation_root, curation_version, *table_rejection)
        return []
    saved_dirs = []
    for col_idx, col in enumerate(list(pq_df.columns)):
        col_y = pq_df[col]
        is_cls = _infer_is_cls(col_y)
        if is_cls and not need_cls:
            continue
        if not is_cls and not need_reg:
            continue
        # Column text itself is unbounded (some T4 tables have 150+ char names) and would blow
        # past the filesystem's filename length limit; an enumerated index keeps this bounded.
        # The original column name is still recorded in CurationOutcome.column.
        name = _sanitize_name(f"{pq_name}__col{col_idx}")
        if col_y.isnull().values.any():
            if _cached_column_outcome(curation_root, curation_version, is_cls, name) is None:
                path = _curation_path(curation_root, curation_version, is_cls, name, kept=False)
                CurationOutcome(name=name, parquet_file=pq_name, column=col, is_cls=is_cls, kept=False,
                                reject_reason=REASON_NULL_TARGET, score=None, train_size=None).save(path)
            continue
        data_dir = join(output_dir, "cls" if is_cls else "reg", name)
        if exists(join(data_dir, RealDataset.PROPERTIES)):
            # Already materialized and already counted once by _existing_materialized_dirs
            # at the top of create_t4_data; re-appending here would double-count it every
            # time a resumed run rescans the chunk that produced it.
            continue
        try:
            created = _materialize_column(pq_df=pq_df, col=col, name=name, pq_name=pq_name, is_cls=is_cls,
                                          data_dir=data_dir, d_num=d_num, curation_root=curation_root,
                                          curation_version=curation_version)
        except Exception as e:
            print(f"⚠️ Skipping {name}: {type(e).__name__}: {e}")
            continue
        if created is not None:
            saved_dirs.append(created)
    return saved_dirs


def _reject_table(pq_df: DataFrame, pq_name: str, curation_root: str, curation_version: str, reject_reason: str,
                  duplicate_of: str) -> None:
    for col_idx, col in enumerate(list(pq_df.columns)):
        is_cls = _infer_is_cls(pq_df[col])
        name = _sanitize_name(f"{pq_name}__col{col_idx}")
        accepted_path = _curation_path(curation_root, curation_version, is_cls, name, kept=True)
        if exists(accepted_path):
            shutil.rmtree(os.path.dirname(accepted_path))
        path = _curation_path(curation_root, curation_version, is_cls, name, kept=False)
        CurationOutcome(name=name, parquet_file=pq_name, column=col, is_cls=is_cls, kept=False,
                        reject_reason=reject_reason, score=None, train_size=None, duplicate_of=duplicate_of).save(path)


def _existing_materialized_dirs(output_dir: str) -> List[str]:
    """Datasets already fully materialized in a prior run, so re-running with the same
    output_dir picks up where it left off instead of recomputing from scratch."""
    dirs = []
    for task in ("cls", "reg"):
        task_dir = join(output_dir, task)
        if not exists(task_dir):
            continue
        for name in os.listdir(task_dir):
            d = join(task_dir, name)
            if exists(join(d, RealDataset.PROPERTIES)):
                dirs.append(d)
    return dirs


def _infer_is_cls(y: Series) -> bool:
    """Non-numeric dtypes are always classification; low-cardinality numeric columns (encoded
    categories) are treated as classification too, rather than left to fail the regression
    unique-value floor and be discarded.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return True
    return y.nunique() <= MAX_CLS_UNIQUE


def _curation_dir(curation_root: str, curation_version: str, is_cls: bool) -> str:
    task = "cls" if is_cls else "reg"
    return join(curation_root, f"t4_curation_{task}_{curation_version}")


def _curation_path(curation_root: str, curation_version: str, is_cls: bool, name: str, kept: bool) -> str:
    verdict = "accepted" if kept else "rejected"
    return join(_curation_dir(curation_root, curation_version, is_cls), verdict, name, CURATION_FILE)


def _find_curation_path(curation_root: str, curation_version: str, is_cls: bool, name: str) -> Optional[str]:
    for kept in (True, False):
        path = _curation_path(curation_root, curation_version, is_cls, name, kept=kept)
        if exists(path):
            return path
    return None


def _cached_column_outcome(curation_root: str, curation_version: str, is_cls: bool,
                           name: str) -> Optional[CurationOutcome]:
    """The cached verdict of a table that reached curation, discarding one left by a table-level rejection."""
    path = _find_curation_path(curation_root, curation_version, is_cls, name)
    if path is None:
        return None
    outcome = CurationOutcome.from_json(path)
    if outcome.reject_reason in TABLE_REJECT_REASONS:
        shutil.rmtree(os.path.dirname(path))
        return None
    return outcome


def count_curated(curation_root: str, curation_version: str, is_cls: bool) -> Tuple[int, int]:
    """(n_accepted, n_rejected) so far for this task/version: a single directory listing each,
    no JSON parsing, so it stays fast even once curation.json files number in the hundreds of
    thousands."""
    base = _curation_dir(curation_root, curation_version, is_cls)
    def _count(verdict: str) -> int:
        verdict_dir = join(base, verdict)
        return len(os.listdir(verdict_dir)) if exists(verdict_dir) else 0
    return _count("accepted"), _count("rejected")


def _materialize_column(pq_df: DataFrame, col: str, name: str, pq_name: str, is_cls: bool, data_dir: str,
                        d_num: int, curation_root: str, curation_version: str) -> Optional[str]:
    """Reuses a cached curation verdict when one exists; otherwise evaluates and caches it. The
    (cheap) verbalize-and-save step below always runs from raw columns, independent of the cache,
    so a preprocessing change only redoes this step and not the (expensive) LightGBM filtering.
    """
    col_y = pq_df[col]
    col_x = pq_df.drop(columns=col)
    outcome = _cached_column_outcome(curation_root, curation_version, is_cls, name)
    if outcome is None:
        outcome = _evaluate_column(x=col_x, y=col_y, is_cls=is_cls, name=name, parquet_file=pq_name, column=col)
        outcome.save(_curation_path(curation_root, curation_version, is_cls, name, kept=outcome.kept))
    if not outcome.kept:
        return None
    preprocessor = TabSTARv1p1Verbalizer(is_cls=is_cls, d_num=d_num)
    preprocessor.fit(col_x, col_y)
    train_data = preprocessor.transform(col_x, col_y)
    properties = RealDatasetProperties(name=name, d_output=train_data.d_output, train_size=len(train_data))
    save_pretrain_dataset(data_dir=data_dir, train_data=train_data, properties=properties)
    return data_dir


def _evaluate_column(x: DataFrame, y: Series, is_cls: bool, name: str, parquet_file: str,
                     column: str) -> CurationOutcome:
    skip_reason = evaluate_skip_reason(x=x, y=y, is_cls_task=is_cls)
    if skip_reason is not None:
        return CurationOutcome(name=name, parquet_file=parquet_file, column=column, is_cls=is_cls, kept=False,
                               reject_reason=skip_reason, score=None, train_size=None)
    score, reject_reason = evaluate_cls_task(x=x, y=y) if is_cls else evaluate_regression_task(x=x, y=y)
    kept = reject_reason is None
    return CurationOutcome(name=name, parquet_file=parquet_file, column=column, is_cls=is_cls, kept=kept,
                           reject_reason=reject_reason, score=score, train_size=len(x) if kept else None)


def _pretty_parquet_name(pq_f: str, hf_cache_dir: str) -> str:
    name = pq_f
    for bad in [hf_cache_dir.rstrip('/') + "/", ".parquet", "="]:
        name = name.replace(bad, "")
    return name.replace('-', '_').replace('/', '__')


def _sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.]+', '_', name)


def evaluate_skip_reason(x: DataFrame, y: Series, is_cls_task: bool) -> Optional[str]:
    if len(x) < MIN_PRETRAIN_ROWS:
        return REASON_TOO_FEW_ROWS
    assert len(x) <= MAX_PRETRAIN_ROWS, f"{len(x)=} exceeds {MAX_PRETRAIN_ROWS=}"
    if len(x.columns) < MIN_PRETRAIN_FEATURES:
        return REASON_TOO_FEW_FEATURES
    if len(x.columns) > MAX_PRETRAIN_FEATURES:
        return REASON_TOO_MANY_FEATURES
    y_unique = len(y.unique())
    if is_cls_task:
        if y_unique > MAX_CLS_UNIQUE:
            return REASON_TOO_MANY_CLASSES
        if min(y.value_counts()) < MIN_CLS_EXAMPLES:
            return REASON_RARE_CLASS
    if (not is_cls_task) and (y_unique < MIN_REGRESSION_UNIQUE):
        return REASON_TOO_FEW_UNIQUE_REGRESSION
    return None


def evaluate_regression_task(x: DataFrame, y: Series) -> Tuple[Optional[float], Optional[str]]:
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=False)
    try:
        with silence_baselines_prints():
            model = LightGBM(problem_type=SupervisedTask.REGRESSION, device=torch.device("cpu"),
                              num_threads=LGBM_NUM_THREADS)
            model.fit(x_train, y_train)
    except LightGBMError:
        return None, REASON_LIGHTGBM_ERROR
    score = model.score_all_metrics(X=x_test, y=y_test).score
    if score < MIN_RSQ:
        return score, REASON_SCORE_TOO_LOW
    if score > MAX_RSQ:
        return score, REASON_SCORE_TOO_HIGH
    return score, None


def evaluate_cls_task(x: DataFrame, y: Series) -> Tuple[Optional[float], Optional[str]]:
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=True)
    if set(y_test).difference(set(y_train)):
        return None, REASON_UNSEEN_TEST_CLASS
    problem_type = SupervisedTask.BINARY if len(set(y_train)) == 2 else SupervisedTask.MULTICLASS
    try:
        with silence_baselines_prints():
            model = LightGBM(problem_type=problem_type, device=torch.device("cpu"),
                              num_threads=LGBM_NUM_THREADS)
            model.fit(x_train, y_train)
    except LightGBMError:
        return None, REASON_LIGHTGBM_ERROR
    score = model.score_all_metrics(X=x_test, y=y_test).score
    if score < MIN_AUC:
        return score, REASON_SCORE_TOO_LOW
    if score > MAX_AUC:
        return score, REASON_SCORE_TOO_HIGH
    return score, None
