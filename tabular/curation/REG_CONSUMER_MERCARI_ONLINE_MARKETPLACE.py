from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mercari_price_suggestion100K
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46660
====
Description: Predict the price of items sold in the online marketplace of Mercari based on
    information from the product page like name, description, free shipping availability, etc.
    This data originates from a 2017 Kaggle competition (https://www.kaggle.com/c/
    mercari-price-suggestion-challenge/), in which 1st place and 3rd place engineered
    dataset-specific text features such as customized bag-of-words and character N-grams, carefully
    tuned learning-rate/batch-size schedules, and specially ensembled models in a dataset-specific manner.
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: log_price (numeric, 415 distinct): ['2.3979', '2.5649', '2.7081', '2.8332', '2.3026', '2.1972', '2.7726', '3.0445', '2.0794', '3.2189']
====
Features:

train_id (numeric, 100000 distinct): ['1173576', '567230', '999013', '1166536', '567382', '349984', '1083634', '1465833', '321483', '99430']
name (string, 94033 distinct): ['Bundle', 'BUNDLE', 'Reserved', 'Lularoe TC leggings', 'Dress', 'Coach purse', 'Vans', 'Converse', 'Nike', 'Romper']
item_condition_id (numeric, 5 distinct): ['1', '3', '2', '4', '5']
category_name (string, 988 distinct): ['Women/Athletic Apparel/Pants, Tights, Leggings', 'Women/Tops & Blouses/T-Shirts', 'Beauty/Makeup/Face', 'Beauty/Makeup/Lips', 'Electronics/Video Games & Consoles/Games', 'Beauty/Makeup/Eyes', 'Electronics/Cell Phones & Accessories/Cases, Covers & Skins', 'Women/Underwear/Bras', 'Women/Tops & Blouses/Tank, Cami', 'Women/Tops & Blouses/Blouse']
brand_name (string, 2024 distinct): ['Nike', 'PINK', "Victoria's Secret", 'LuLaRoe', 'Apple', 'Lululemon', 'FOREVER 21', 'Nintendo', 'Michael Kors', 'American Eagle']
price (numeric, 415 distinct): ['10.0', '12.0', '14.0', '16.0', '9.0', '8.0', '15.0', '20.0', '7.0', '24.0']
shipping (numeric, 2 distinct): ['0', '1']
item_description (string, 90584 distinct): ['No description yet', 'New', 'Brand new', 'Good condition', 'Great condition', 'Like new', 'Never worn', 'Excellent condition', 'Never used', 'NWT']
cat1 (string, 11 distinct): ['Women', 'Beauty', 'Kids', 'Electronics', 'Men', 'Home', 'Vintage & Collectibles', 'Other', 'Handmade', 'Sports & Outdoors']
cat2 (string, 114 distinct): ['Athletic Apparel', 'Makeup', 'Tops & Blouses', 'Shoes', 'Jewelry', 'Toys', 'Cell Phones & Accessories', "Women's Handbags", 'Dresses', "Women's Accessories"]
cat3 (string, 706 distinct): ['Pants, Tights, Leggings', 'Other', 'Face', 'T-Shirts', 'Shoes', 'Games', 'Lips', 'Athletic', 'Eyes', 'Cases, Covers & Skins']
'''

CONTEXT = "Mercari Online Marketplace Product Prices"
TARGET = CuratedTarget(raw_name="log_price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["train_id", "price"]
FEATURES = []
