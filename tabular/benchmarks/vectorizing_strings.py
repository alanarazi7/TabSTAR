from tabular.datasets.tabular_datasets import KaggleDatasetID, UrlDatasetID, OpenMLDatasetID

# Vectorizing string entries for data processing on tables: when are larger language models better?
# https://arxiv.org/pdf/2312.09634


# couldn't find 'journal influence' or 'us presidential'
VECTORIZING = [
    OpenMLDatasetID.MUL_FOOD_WINE_REVIEW,
    KaggleDatasetID.MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23,
    UrlDatasetID.REG_CONSUMER_BIKE_PRICE_BIKEWALE,
    KaggleDatasetID.REG_FOOD_RAMEN_RATINGS_2022,
    KaggleDatasetID.REG_FOOD_ZOMATO_RESTAURANTS,
    UrlDatasetID.REG_SOCIAL_BOOKS_GOODREADS,
    KaggleDatasetID.REG_SOCIAL_BOOK_READABILITY_CLEAR,
    KaggleDatasetID.REG_SOCIAL_SPOTIFY_POPULARITY,
    KaggleDatasetID.REG_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE,
    UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER,
    OpenMLDatasetID.REG_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY,
    ]
