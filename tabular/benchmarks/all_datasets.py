from tabular.datasets.tabular_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID


# Datasets with too many features
TOO_MANY_FEATURES = [
    OpenMLDatasetID.REG_ANONYM_MERCEDES_BENZ_GREENER_MANUFACTURING,  # 376
    OpenMLDatasetID.REG_ANONYM_SANTANDER_TRANSACTION_VALUE,          # 4991
    OpenMLDatasetID.REG_ANONYM_TOPO,                                 # 266
    OpenMLDatasetID.REG_ANONYM_YPROP,                                # 251
    OpenMLDatasetID.REG_SCIENCE_QSAR_TID_10980,                      # 1024
    OpenMLDatasetID.REG_SCIENCE_QSAR_TID_11,                         # 1024
]
