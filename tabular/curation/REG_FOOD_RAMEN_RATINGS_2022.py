from typing import Optional

from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ankanhore545/top-ramen-ratings-2022/Top Ramen Ratings .csv
====
Examples: 4120
====
URL: https://www.kaggle.com/ankanhore545/top-ramen-ratings-2022/Top Ramen Ratings .csv
====
Description: 
Top Ramen Ratings 2022
Best Ramen Ratings Globally ( 2022)

About Dataset
The list helps to analyse and check the ramen ratings from the best places in the world.

The dataset has been extracted from this site
Inspiration: The food reviews and ranks would definitely help to understand the food joints, locations, the quality and how popular those ramen places are.

This would also affect the small and medium outlets, thereby effecting the economy who are working in these chains to earn their incomes.

====
Features:

Review # (int64, 4120 distinct): ['4120', '1416', '1382', '1381', '1380', '1379', '1378', '1377', '1376', '1375']
Brand (object, 616 distinct): ['Nissin', 'Maruchan', 'Myojo', 'Nongshim', 'Samyang Foods', 'Paldo', 'Sapporo Ichiban', 'Mama', 'Acecook', 'Indomie']
Variety (object, 3828 distinct): ['Miso Ramen', 'Yakisoba', 'Beef', 'Chicken', 'Vegetable', 'Instant Noodles Chicken Flavour', 'Artificial Chicken', 'Tempura Soba', 'Instant Noodles Beef Flavour', 'Tonkotsu Ramen']
Style (object, 9 distinct): ['Pack', 'Bowl', 'Cup', 'Tray', 'Box', 'Restaurant', 'Bottle', 'Can', 'Bar']
Country (object, 53 distinct): ['Japan', 'United States', 'South Korea', 'Taiwan', 'China', 'Thailand', 'Malaysia', 'Hong Kong', 'Indonesia', 'Singapore']
Stars (object, 50 distinct): ['5', '3.5', '3.75', '4', '4.5', '3.25', '4.25', '3', '2.75', '2']
T (float64, 0 distinct): []
'''

def fix_ramen_rating(rating: str) -> Optional[float]:
    assert isinstance(rating, str)
    for c in rating:
        if c not in "0123456789.":
            return None
    return float(rating)

CONTEXT = "Ramen Ratings"
TARGET = CuratedTarget(raw_name="Stars", task_type=SupervisedTask.REGRESSION,
                       processing_func=fix_ramen_rating)
COLS_TO_DROP = ["T", "Review #"]
FEATURES = []

DESCRIPTION = '''
Top Ramen Ratings 2022
Best Ramen Ratings Globally ( 2022)

About Dataset
The list helps to analyse and check the ramen ratings from the best places in the world.

The dataset has been extracted from this site
Inspiration: The food reviews and ranks would definitely help to understand the food joints, locations, the quality and how popular those ramen places are.

This would also affect the small and medium outlets, thereby effecting the economy who are working in these chains to earn their incomes.
'''