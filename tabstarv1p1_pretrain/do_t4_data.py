import os

# Must run before any other import: torch/numpy/lightgbm size their thread pools at
# library-load time, and `tabstarv1p1_pretrain.pretraining.hdf5` pulls in torch before
# `tabstarv1p1_pretrain.pretraining.t4_data` gets a chance to set these itself.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse

import tabstarv1p1_pretrain.constants  # noqa: F401, loads .env
from tabstarv1p1_pretrain.pretraining.hdf5 import RealDatasetProperties
from tabstarv1p1_pretrain.pretraining.t4_data import DEFAULT_HF_CACHE_DIR, DEFAULT_OUTPUT_DIR, create_t4_data
from tabstarv1p1_pretrain.pretraining.arguments import D_NUM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare T4/TabLib-derived pretrain datasets.")
    parser.add_argument('--hf_cache_dir', type=str, default=DEFAULT_HF_CACHE_DIR,
                        help="Directory holding the extracted (or still-zipped) chunk-XXXX T4 folders. "
                             "Downloaded here automatically if missing.")
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Where prepared RealDataset dirs get written.")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument('--max_datasets', type=int, default=None,
                              help="Stop once this many datasets (cls + reg combined) are saved.")
    target_group.add_argument('--max_per_task', type=int, default=None,
                              help="Stop each task type independently once it reaches this many datasets, "
                                   "skipping further evaluation of a task type that already has enough.")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--d_num', type=int, default=D_NUM)
    parser.add_argument('--curation_root', type=str, default="../t4_curation",
                        help="Parent dir for the versioned t4_curation_{cls,reg}_<curation_version> cache folders.")
    parser.add_argument('--curation_version', type=str, default="v1",
                        help="Cache is keyed by this; bump it to force re-running the LightGBM filtering.")
    parser.add_argument('--eval_fingerprints', type=str, default=None,
                        help="JSON from do_t4_eval_fingerprints.py; T4 tables copying one of its tables are dropped.")
    args = parser.parse_args()
    if args.max_datasets is None and args.max_per_task is None:
        args.max_datasets = 10_000

    saved = create_t4_data(hf_cache_dir=args.hf_cache_dir, output_dir=args.output_dir, d_num=args.d_num,
                           max_datasets=args.max_datasets, max_per_task=args.max_per_task, workers=args.workers,
                           curation_root=args.curation_root, curation_version=args.curation_version,
                           eval_fingerprints=args.eval_fingerprints)
    n_cls = sum(RealDatasetProperties.from_json(d).is_cls for d in saved)
    n_reg = len(saved) - n_cls
    print(f"🌟 Prepared {len(saved)} T4 datasets under {args.output_dir} ({n_reg} reg, {n_cls} cls)")
