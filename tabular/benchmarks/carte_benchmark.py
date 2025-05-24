from tabular.datasets.tabular_datasets import KaggleDatasetID, UrlDatasetID, OpenMLDatasetID

# https://huggingface.co/datasets/inria-soda/carte-benchmark
# https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository


# Removed "BUY_BUY_BABY" as it is roughly the same as "BABIES_R_US_PRICES",
# Also "BIKDEKHO" is the same as "BIKEWALE"
# Also clarivate "Journal Score JCR" as we have "REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_CITATIONS"
# "Japanese Anime" seemed too similar to "Anime Planet"
# "Mydramalist" is too similar to "Korean Drama" (seems to overlap)
# "Prescription Drugs" was skipped because of problems accessing
# "Roger Ebert" was very slow / problematic to download with pkl, skipping
# US Presidential was not comfortable from github
# "Used Cars 24" might be too similar to others
# "Whisky" might be too similar to REG_GOOD_WHISKY_SCOTCH_REVIEWS
# Many wine datasets were skipped as we have MUL_FOOD_WINE_REVIEW
CARTE_BENCHMARK = [
    KaggleDatasetID.MUL_FOOD_MICHELIN_GUIDE_RESTAURANTS,
    OpenMLDatasetID.MUL_FOOD_WINE_REVIEW,
    KaggleDatasetID.MUL_FOOD_YELP_REVIEWS,
    KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23,
    UrlDatasetID.REG_CONSUMER_BABIES_R_US_PRICES,
    UrlDatasetID.REG_CONSUMER_BIKE_PRICE_BIKEWALE,
    KaggleDatasetID.REG_CONSUMER_CAR_PRICE_CARDEKHO,
    KaggleDatasetID.REG_FOOD_ALCOHOL_WIKILIQ_PRICES,
    KaggleDatasetID.REG_FOOD_BEER_RATINGS,
    KaggleDatasetID.REG_FOOD_CHOCOLATE_BAR_RATINGS,
    KaggleDatasetID.REG_FOOD_COFFEE_REVIEW,
    KaggleDatasetID.REG_FOOD_ZOMATO_RESTAURANTS,
    KaggleDatasetID.REG_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE,
    UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER,
    OpenMLDatasetID.REG_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY,
    UrlDatasetID.REG_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES,
    UrlDatasetID.REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT,
    KaggleDatasetID.REG_SOCIAL_ANIME_PLANET_RATING,
    KaggleDatasetID.REG_SOCIAL_FILMTV_MOVIE_RATING_ITALY,
    KaggleDatasetID.REG_SOCIAL_BOOK_READABILITY_CLEAR,
    KaggleDatasetID.REG_SOCIAL_KOREAN_DRAMA,
    KaggleDatasetID.REG_SOCIAL_MOVIES_DATASET_REVENUE,
    UrlDatasetID.REG_SOCIAL_MOVIES_ROTTEN_TOMATOES,
    KaggleDatasetID.REG_SOCIAL_MUSEUMS_US_REVENUES,
    KaggleDatasetID.REG_SOCIAL_VIDEO_GAMES_SALES,
    OpenMLDatasetID.REG_SPORTS_FIFA22_WAGES,
    KaggleDatasetID.REG_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT,
    KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY,
    KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_PAKISTAN,
    KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_SAUDI_ARABIA,
    KaggleDatasetID.REG_FOOD_WINE_POLISH_MARKET_PRICES,
    KaggleDatasetID.REG_FOOD_RAMEN_RATINGS_2022,
    KaggleDatasetID.REG_FOOD_WINE_VIVINO_SPAIN,
                   ]