from importlib import import_module
from pkgutil import iter_modules

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar_paper.datasets import annotated
from tabstar_paper.datasets.curation_objects import CuratedDataset

CURATIONS = {}


def construct_curation_dict():
    modules = [m for m in iter_modules(annotated.__path__, "tabstar_paper.datasets.annotated.") if not m.ispkg]
    for m in modules:
        module = import_module(m.name)
        sid = m.name.split('.')[-1]
        curated = CuratedDataset.from_module(module)
        CURATIONS[sid] = curated

def get_curated(dataset_id: TabularDatasetID) -> CuratedDataset:
    return CURATIONS[dataset_id.name]


construct_curation_dict()
