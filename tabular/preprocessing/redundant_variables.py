from pandas import DataFrame

from tabular.datasets.manual_curation_obj import CuratedDataset


def drop_redundant_columns(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    if curation.cols_to_drop:
        x = x.drop(columns=curation.cols_to_drop, errors='ignore')
    return x