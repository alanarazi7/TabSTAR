from importlib import import_module
from pkgutil import iter_modules

from tabular.datasets.manual_curation_obj import CuratedDataset
from tabular import curation
from tabular.datasets.tabular_datasets import TabularDatasetID

CURATIONS = {}


def construct_curation_dict():
    modules = [m for m in iter_modules(curation.__path__, "tabular.curation.") if not m.ispkg]
    for m in modules:
        module = import_module(m.name)
        sid = m.name.split('.')[-1]
        curated = CuratedDataset.from_module(module)
        CURATIONS[sid] = curated

def get_curated(dataset_id: TabularDatasetID) -> CuratedDataset:
    return CURATIONS[dataset_id.name]


construct_curation_dict()
