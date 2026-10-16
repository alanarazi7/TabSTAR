"""Fingerprints the evaluation tables that T4 pretrain data must not contain.

TabArena tables come from OpenML in full, target included, since a scraped copy carries it too.
BeyondArena and STRABLE tables come from every source parquet in their HF repos, streamed because
some are several GB. MulTaBench tables are each dataset's Kaggle `data.csv`, images excluded. The
output JSON is what `do_t4_data.py --eval_fingerprints` blocks against.
"""
import argparse
import json
from typing import Dict, List, Optional

import tabstarv1p1_pretrain.constants  # noqa: F401, loads .env
from tabstarv1p1_pretrain.pretraining.t4_overlap import SKETCH_SIZE, frame_signatures, parquet_signatures

BEYONDARENA_REPO_ID = "TabArena/BeyondArena"
STRABLE_REPO_ID = "inria-soda/STRABLE-benchmark"
STRABLE_REVISION = "335c0a77595d4a9ddd86f46dc9c3b414f63a09a8"
MULTABENCH_KAGGLE_OWNER = "chico89"
# Core image, core text and full-text extra lists of alanarazi7/MulTaBench.
MULTABENCH_SLUGS = (
    "multabench-celeb-attractiveness", "multabench-hateful-meme", "multabench-mammography-cmmd",
    "multabench-chexpert", "multabench-cbis-ddsm", "multabench-glaucoma-smdg", "multabench-csgo-skin",
    "multabench-flower-bouquets", "multabench-hubmap-hpa", "multabench-justin-instagram", "multabench-petfinder",
    "multabench-zooscan-zooplankton", "multabench-amazon-bestseller", "multabench-amazon-packages",
    "multabench-hnm-fashion", "multabench-khaadi-clothes", "multabench-letterboxd-movies", "multabench-mango-mass",
    "multabench-mkphoto-bots", "multabench-painting-price",
    "multabench-fake-job-posting", "multabench-jigsaw-toxicity", "multabench-kickstarter-funding",
    "multabench-data-scientist-salary", "multabench-michelin-restaurants", "multabench-product-sentiment",
    "multabench-spotify-genres", "multabench-us-accidents", "multabench-wine-review",
    "multabench-women-clothing-review", "multabench-babies-prices", "multabench-book-price",
    "multabench-book-readability", "multabench-mercari-marketplace", "multabench-montgomery-salaries",
    "multabench-rotten-tomatoes", "multabench-scimagojr-impact", "multabench-vancouver-salaries",
    "multabench-video-games-sales", "multabench-zomato-restaurants",
    "multabench-full-wine-polish-market", "multabench-full-korean-drama", "multabench-full-used-car-saudi",
    "multabench-full-consumer-complaint", "multabench-full-wine-vivino-spain",
    "multabench-full-chocolate-bar-ratings", "multabench-full-anime-planet", "multabench-full-used-car-pakistan",
    "multabench-full-books-goodreads", "multabench-full-ramen-ratings", "multabench-full-osha-injury",
    "multabench-full-fifa22-wages", "multabench-full-california-prices", "multabench-full-wikiliq-prices",
    "multabench-full-american-eagle-prices", "multabench-full-airbnb-seattle", "multabench-full-news-channel",
    "multabench-full-imdb-genre", "multabench-full-melbourne-airbnb", "multabench-full-box-office",
)
FORMAT_VERSION = "t4-eval-fingerprints-v1"


def _record(table_id: str, source: str, schema: str, sketches: List[List[int]]) -> Dict:
    return {"table_id": table_id, "source": source, "schema": schema, "sketches": sketches}


def fingerprint_tabarena() -> List[Dict]:
    import openml
    from tabarena.benchmark.task.metadata.fetch_metadata import load_curated_task_metadata

    metadata = load_curated_task_metadata().drop_duplicates("dataset_id")
    tables = []
    for i, row in enumerate(metadata.itertuples(), start=1):
        dataset = openml.datasets.get_dataset(int(row.dataset_id), download_data=True)
        frame, _, _, _ = dataset.get_data(dataset_format="dataframe")
        schema, sketches = frame_signatures(frame)
        tables.append(_record(f"tabarena:{row.dataset_name}", "tabarena", schema, sketches))
        print(f"[{i}/{len(metadata)}] tabarena {row.dataset_name} ({len(sketches)} sketched columns)")
    return tables


def _fingerprint_hf_parquets(source: str, repo_id: str, filename: str, revision: Optional[str] = None) -> List[Dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset", revision=revision)
                   if f.endswith(f"/{filename}"))
    tables = []
    for i, remote in enumerate(files, start=1):
        path = hf_hub_download(repo_id, remote, repo_type="dataset", revision=revision)
        schema, sketches = parquet_signatures(path)
        table_name = remote.split("/")[0]
        tables.append(_record(f"{source}:{table_name}", source, schema, sketches))
        print(f"[{i}/{len(files)}] {source} {table_name} ({len(sketches)} sketched columns)")
    return tables


def fingerprint_beyondarena() -> List[Dict]:
    return _fingerprint_hf_parquets("beyondarena", BEYONDARENA_REPO_ID, "dataset.parquet")


def fingerprint_strable() -> List[Dict]:
    return _fingerprint_hf_parquets("strable", STRABLE_REPO_ID, "data.parquet", revision=STRABLE_REVISION)


def fingerprint_multabench() -> List[Dict]:
    import kagglehub
    import pandas as pd

    tables = []
    for i, slug in enumerate(MULTABENCH_SLUGS, start=1):
        path = kagglehub.dataset_download(f"{MULTABENCH_KAGGLE_OWNER}/{slug}", path="data.csv")
        schema, sketches = frame_signatures(pd.read_csv(path))
        tables.append(_record(f"multabench:{slug}", "multabench", schema, sketches))
        print(f"[{i}/{len(MULTABENCH_SLUGS)}] multabench {slug} ({len(sketches)} sketched columns)")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint the evaluation benchmarks for T4 decontamination.")
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--skip_tabarena', action='store_true')
    parser.add_argument('--skip_beyondarena', action='store_true')
    parser.add_argument('--skip_strable', action='store_true')
    parser.add_argument('--skip_multabench', action='store_true')
    args = parser.parse_args()

    tables = []
    if not args.skip_tabarena:
        tables.extend(fingerprint_tabarena())
    if not args.skip_beyondarena:
        tables.extend(fingerprint_beyondarena())
    if not args.skip_strable:
        tables.extend(fingerprint_strable())
    if not args.skip_multabench:
        tables.extend(fingerprint_multabench())
    payload = {"format_version": FORMAT_VERSION, "sketch_size": SKETCH_SIZE, "tables": tables}
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"🌟 Fingerprinted {len(tables)} evaluation tables into {args.out}")
