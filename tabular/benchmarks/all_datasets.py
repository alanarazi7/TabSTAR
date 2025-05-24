from tabular.benchmarks.amlb import AMLB
from tabular.benchmarks.carte_benchmark import CARTE_BENCHMARK
from tabular.benchmarks.ctr23 import CTR23_REG
from tabular.benchmarks.grinsztajn import GRINSZTAJN
from tabular.benchmarks.multimodal import MULTIMODAL
from tabular.benchmarks.tabzilla import TABZILLA
from tabular.benchmarks.vectorizing_strings import VECTORIZING
from tabular.datasets.metadata import DATA2EXAMPLES
from tabular.datasets.tabular_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID


# Datasets with too many features
TOO_MANY_FEATURES = [
    OpenMLDatasetID.BIN_ANONYM_ARCENE,                               # 10000
    OpenMLDatasetID.BIN_ANONYM_BIORESPONSE,                          # 1776
    OpenMLDatasetID.BIN_ANONYM_CHRISTINE,                            # 1636
    OpenMLDatasetID.BIN_ANONYM_GINA,                                 # 970
    OpenMLDatasetID.BIN_ANONYM_GUILLERMO,                            # 4296
    OpenMLDatasetID.BIN_ANONYM_KDDCUP_98_DIRECT_MAIL,                # 478
    OpenMLDatasetID.BIN_ANONYM_KDDCUP_09_APPETENCY,                  # 230
    OpenMLDatasetID.BIN_ANONYM_KDDCUP_09_UPSELLING,                  # 14891
    OpenMLDatasetID.BIN_ANONYM_MADELINE,                             # 259
    OpenMLDatasetID.BIN_ANONYM_MADELONE,                             # 500
    OpenMLDatasetID.BIN_ANONYM_PHILIPPINE,                           # 308
    OpenMLDatasetID.BIN_ANONYM_RICARDO,                              # 4296
    OpenMLDatasetID.BIN_CONSUMER_INTERNET_ADVERTISEMENTS,            # 1558
    OpenMLDatasetID.BIN_GENETICS_OVA_BREAST,                         # 10935
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


TEXTUAL_DATASETS = list({d for ls in [MULTIMODAL, CARTE_BENCHMARK, VECTORIZING] for d in ls})

ANALYSIS_TEXT_DOWNSTREAM = [OpenMLDatasetID.REG_SPORTS_FIFA22_WAGES,
                            KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23,
                            KaggleDatasetID.MUL_FOOD_MICHELIN_GUIDE_RESTAURANTS,
                            KaggleDatasetID.REG_FOOD_RAMEN_RATINGS_2022,
                            UrlDatasetID.REG_SOCIAL_BOOKS_GOODREADS,
                            KaggleDatasetID.REG_FOOD_WINE_POLISH_MARKET_PRICES,
                            OpenMLDatasetID.MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW,
                            OpenMLDatasetID.MUL_SOCIAL_NEWS_CHANNEL_CATEGORY,
                            OpenMLDatasetID.REG_CONSUMER_MERCARI_ONLINE_MARKETPLACE,
                            KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY,
                            KaggleDatasetID.REG_SOCIAL_VIDEO_GAMES_SALES,
                            UrlDatasetID.REG_CONSUMER_BABIES_R_US_PRICES,
                            OpenMLDatasetID.REG_CONSUMER_BOOK_PRICE_PREDICTION,
                            KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_PAKISTAN,
                            KaggleDatasetID.REG_SOCIAL_ANIME_PLANET_RATING,
                            OpenMLDatasetID.MUL_FOOD_WINE_REVIEW,
                            OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT,
                            OpenMLDatasetID.MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY,
                            KaggleDatasetID.REG_SOCIAL_MUSEUMS_US_REVENUES,
                            OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING, ]


TEXTUAL_BIG = [d for d, n in DATA2EXAMPLES.items() if n > 10_000 and d in TEXTUAL_DATASETS]

BENCHMARKS2DATASETS = {'AMLB': AMLB,
                       'CTR23': CTR23_REG,
                       'GRINSZTAJN': GRINSZTAJN,
                       'MULTIMODAL': MULTIMODAL,
                       'VECTORIZING': VECTORIZING,
                       'CARTE': CARTE_BENCHMARK,
                       'TABZILLA': TABZILLA}
