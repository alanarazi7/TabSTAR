from typing import Optional

from tabstar.datasets.all_datasets import KaggleDatasetID, UrlDatasetID, OpenMLDatasetID, TabularDatasetID
from tabstar.datasets.pretrain_folds import PRETRAIN2FOLD

TEXT2FOLD = {KaggleDatasetID.REG_SOCIAL_ANIME_PLANET_RATING: 0,
             KaggleDatasetID.REG_FOOD_RAMEN_RATINGS_2022: 1,
             UrlDatasetID.REG_CONSUMER_BIKE_PRICE_BIKEWALE: 2,
             KaggleDatasetID.REG_SOCIAL_BOOK_READABILITY_CLEAR: 3,
             KaggleDatasetID.REG_FOOD_ZOMATO_RESTAURANTS: 4,
             OpenMLDatasetID.REG_CONSUMER_JC_PENNEY_PRODUCT_PRICE: 0,
             KaggleDatasetID.REG_SOCIAL_FILMTV_MOVIE_RATING_ITALY: 1,
             KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_PAKISTAN: 2,
             OpenMLDatasetID.MUL_SOCIAL_GOOGLE_QA_TYPE_REASON: 3,
             UrlDatasetID.REG_SOCIAL_MOVIES_ROTTEN_TOMATOES: 4,
             OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT: 0,
             OpenMLDatasetID.REG_CONSUMER_AMERICAN_EAGLE_PRICES: 1,
             OpenMLDatasetID.MUL_HOUSES_MELBOURNE_AIRBNB: 2,
             KaggleDatasetID.REG_SOCIAL_VIDEO_GAMES_SALES: 3,
             KaggleDatasetID.REG_CONSUMER_CAR_PRICE_CARDEKHO: 4,
             OpenMLDatasetID.MUL_SOCIAL_NEWS_CHANNEL_CATEGORY: 0,
             OpenMLDatasetID.REG_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY: 1,
             OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING: 2,
             KaggleDatasetID.REG_SOCIAL_MOVIES_DATASET_REVENUE: 3,
             KaggleDatasetID.REG_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT: 4,
             OpenMLDatasetID.REG_HOUSES_CALIFORNIA_PRICES_2020: 0,
             OpenMLDatasetID.MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW: 1,
             KaggleDatasetID.MUL_FOOD_MICHELIN_GUIDE_RESTAURANTS: 2,
             KaggleDatasetID.REG_FOOD_WINE_POLISH_MARKET_PRICES: 3,
             OpenMLDatasetID.MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY: 4,
             UrlDatasetID.REG_CONSUMER_BABIES_R_US_PRICES: 0,
             OpenMLDatasetID.REG_SPORTS_FIFA22_WAGES: 1,
             KaggleDatasetID.REG_FOOD_ALCOHOL_WIKILIQ_PRICES: 2,
             KaggleDatasetID.REG_FOOD_CHOCOLATE_BAR_RATINGS: 3,
             UrlDatasetID.REG_SOCIAL_BOOKS_GOODREADS: 4,
             OpenMLDatasetID.BIN_SOCIAL_JIGSAW_TOXICITY: 0,
             KaggleDatasetID.REG_FOOD_WINE_VIVINO_SPAIN: 1,
             KaggleDatasetID.REG_FOOD_COFFEE_REVIEW: 2,
             KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY: 3,
             UrlDatasetID.REG_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES: 4,
             KaggleDatasetID.REG_SOCIAL_MUSEUMS_US_REVENUES: 0,
             OpenMLDatasetID.MUL_FOOD_WINE_REVIEW: 1,
             OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION: 2,
             OpenMLDatasetID.REG_CONSUMER_BOOK_PRICE_PREDICTION: 3,
             OpenMLDatasetID.REG_CONSUMER_MERCARI_ONLINE_MARKETPLACE: 4,
             UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER: 0,
             KaggleDatasetID.REG_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE: 1,
             KaggleDatasetID.REG_SOCIAL_KOREAN_DRAMA: 2,
             KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_SAUDI_ARABIA: 3,
             OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING: 4,
             KaggleDatasetID.REG_SOCIAL_SPOTIFY_POPULARITY: 0,
             KaggleDatasetID.MUL_FOOD_YELP_REVIEWS: 1,
             KaggleDatasetID.REG_FOOD_BEER_RATINGS: 2,
             KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23: 3,
             UrlDatasetID.REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT: 4}


def get_tabstar_version(pretrain_dataset: Optional[TabularDatasetID] = None) -> str:
    """
    Returns the TabStar version based on the pretrain dataset.
    """
    if pretrain_dataset is None:
        return "TabSTAR"

    text_data_folds = {d.name: k for d, k in TEXT2FOLD.items()}
    text_fold = text_data_folds.get(pretrain_dataset.name)
    if text_fold is not None:
        return f"TabSTAR-paper-version-fold-k{text_fold}"

    all_data_folds = {d.name: k for d, k in PRETRAIN2FOLD.items()}
    pretrain_fold = all_data_folds.get(pretrain_dataset.name)
    if pretrain_fold is not None:
        return f"TabSTAR-eval-320-version-fold-k{pretrain_fold}"

    raise ValueError(f"Unknown dataset: {pretrain_dataset}")