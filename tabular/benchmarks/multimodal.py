from tabular.datasets.tabular_datasets import OpenMLDatasetID

# https://github.com/sxjscience/automl_multimodal_benchmark
# We take their 18 datasets. We treat as multiclass two problems that are 'ordinal' ranking but low cardinality
# 'MUL_SOCIAL_GOOGLE_QA_TYPE_REASON' has two flavors  in benchmark, we take only one (it's the same dataset)
# 'REG_CONSUMER_ONLINE_NEWS_POPULARITY' was wrongly used for training, and it was too late to include
MULTIMODAL = [OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
              OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING,
              OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING,
              OpenMLDatasetID.BIN_SOCIAL_JIGSAW_TOXICITY,
              OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT,
              OpenMLDatasetID.MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW,
              OpenMLDatasetID.MUL_FOOD_WINE_REVIEW,
              OpenMLDatasetID.MUL_HOUSES_MELBOURNE_AIRBNB,
              OpenMLDatasetID.MUL_SOCIAL_GOOGLE_QA_TYPE_REASON,
              OpenMLDatasetID.MUL_SOCIAL_NEWS_CHANNEL_CATEGORY,
              OpenMLDatasetID.MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY,
              OpenMLDatasetID.REG_CONSUMER_AMERICAN_EAGLE_PRICES,
              OpenMLDatasetID.REG_CONSUMER_BOOK_PRICE_PREDICTION,
              OpenMLDatasetID.REG_CONSUMER_JC_PENNEY_PRODUCT_PRICE,
              OpenMLDatasetID.REG_CONSUMER_MERCARI_ONLINE_MARKETPLACE,
              OpenMLDatasetID.REG_HOUSES_CALIFORNIA_PRICES_2020]
