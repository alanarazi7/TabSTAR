from tabular.datasets.manual_curation_mapping import CURATIONS, get_curated
from tabular.datasets.tabular_datasets import ALL_DATASETS, get_sid


def test_curation_exists():
    for d in ALL_DATASETS:
        try:
            get_curated(d)
        except KeyError:
            raise NotImplementedError(f"🚨🚨🚨 Dataset {d} is not curated! We must add context")

