from tabular.datasets.tabular_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID


# Datasets with too many features
TOO_MANY_FEATURES = [
    OpenMLDatasetID.BIN_HEALTHCARE_ALZHEIMER_HANDWRITE_DARWIN,       # 450
    OpenMLDatasetID.BIN_PROFESSIONAL_LICD_LABOR_RIGHTS,              # 580
    OpenMLDatasetID.BIN_SCIENCE_HIV_QSAR,                            # 1617
    OpenMLDatasetID.MUL_ANONYM_AMAZON_COMMERCE_REVIEWS,              # 10000
    OpenMLDatasetID.MUL_ANONYM_CNAE,                                 # 856
    OpenMLDatasetID.MUL_ANONYM_DILBERT,                              # 2000
    OpenMLDatasetID.MUL_ANONYM_FABERT,                               # 800
    OpenMLDatasetID.MUL_ANONYM_ISOLET_LETTER_SPEECH_RECOGNITION,     # 617
    OpenMLDatasetID.MUL_ANONYM_MFEAT_FACTORS,                        # 216
    OpenMLDatasetID.MUL_ANONYM_MICRO_MASS,                           # 1300
    OpenMLDatasetID.MUL_ANONYM_ROBERT,                               # 7200
    OpenMLDatasetID.MUL_COMPUTERS_IMAGE_CIFAR10,                     # 1000
    OpenMLDatasetID.MUL_COMPUTERS_IMAGE_GTSRB_GERMAN_TRAFFIC_SIGN,   # 256
    OpenMLDatasetID.MUL_COMPUTERS_IMAGE_INDIAN_PINES,                # 220
    OpenMLDatasetID.MUL_COMPUTERS_IMAGE_MNIST_DIGITS,                # 784
    OpenMLDatasetID.MUL_COMPUTERS_IMAGE_MNIST_FASHION,               # 784
    OpenMLDatasetID.MUL_COMPUTERS_IMAGE_MNIST_JAPANESE_KUZUSHIJI_49, # 784
    OpenMLDatasetID.MUL_HEALTHCARE_HEART_ARRHYTMIA,                  # 279
    OpenMLDatasetID.REG_ANONYM_MERCEDES_BENZ_GREENER_MANUFACTURING,  # 376
    OpenMLDatasetID.REG_ANONYM_SANTANDER_TRANSACTION_VALUE,          # 4991
    OpenMLDatasetID.REG_ANONYM_TOPO,                                 # 266
    OpenMLDatasetID.REG_ANONYM_YPROP,                                # 251
    OpenMLDatasetID.REG_SCIENCE_QSAR_TID_10980,                      # 1024
    OpenMLDatasetID.REG_SCIENCE_QSAR_TID_11,                         # 1024
]
