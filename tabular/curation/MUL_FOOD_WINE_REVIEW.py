from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wine_reviews
====
Examples: 84123
====
URL: https://www.openml.org/search?type=data&id=46653
====
Description: Classify the variety of wines based on tasting descriptions from sommeliers, and numeric
    features like price and categorical features like country-of-origin. The original version of this dataset
    was collected from WineEnthusiast: https://www.kaggle.com/zynicide/wine-reviews
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: variety (string, 30 distinct): ['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend', 'Bordeaux-style Red Blend', 'Riesling', 'Sauvignon Blanc', 'Syrah', 'Rosé', 'Merlot']
====
Features:

country (string, 41 distinct): ['US', 'France', 'Italy', 'Portugal', 'Chile', 'Spain', 'Argentina', 'Austria', 'Australia', 'Germany']
description (string, 78995 distinct): ['Seductively tart in lemon pith, cranberry and pomegranate, this refreshing, light-bodied quaff is infinitely enjoyable, both on its own or at the table. It continues to expand on the palate into an increasing array of fresh flavors, finishing in cherry and orange.', "Ripe plum, game, truffle, leather and menthol are some of the aromas you'll find on this earthy wine. The tightly wound palate offers dried black cherry, chopped sage, mint and roasted coffee bean alongside raspy tannins that leave a mouth-drying finish.", 'Attractive aromas of rose, plum and blue flower float out of the glass. On the palate, vanilla, espresso and toasted oak overwhelm dried black cherry while astringent tannins leave a mouth-drying finish.', 'Pear drop and some citrus are the aromatic markers of this soft, light and very easy-drinking wine. The finish refreshes with green pear flavors.', 'This vineyard is at the heart of the fine run of premier crus that lie midway up the slope above the village. This wine is packed with fruit and also with intense acidity and minerality. It is already refreshing and crisp, although as the toastiness develops, it will become even more impressive. Drink from 2017.', "Easy to drink this blend of all five classic Bordeaux varieties, with Merlot predominating. It's all about cherries, blackberries, plum jam, currants and chocolate. Not particularly subtle or ageworthy, but very pretty.", 'Faint cranberry and incense aromas show on the nose of this bottling. The palate offers more flavors, including coffee, black tar, pomegranate and peppercorns.', 'Enticing aromas like baking spices and a vivid mix of fruit and spice flavors are carried along beautifully by lively acidity matched with luxurious body. This tastes subtly and attractively oaky and has considerable complexity and length.', "Scoury and sweet in the mouth, this blend of Pinot Noir and Chardonnay has forward flavors of peaches, limes, oranges, honey and vanilla. With brisk acidity, it's a nice bubbly to drink now.", "Tough in tannins, with an astringent, lockdown quality. That's the iron fist of this wine. The velvet glove is the rich, flamboyant core of blackberries and cassis. Give this young wine at least six years to come around."]
points (numeric, 21 distinct): ['88', '87', '90', '86', '89', '91', '92', '85', '93', '84']
price (numeric, 343 distinct): ['20.0', '15.0', '25.0', '30.0', '18.0', '40.0', '35.0', '12.0', '50.0', '10.0']
province (string, 362 distinct): ['California', 'Washington', 'Bordeaux', 'Oregon', 'Tuscany', 'Burgundy', 'Mendoza Province', 'Piedmont', 'New York', 'Alsace']
'''

CONTEXT = "Wine Reviews for Variety Prediction"
TARGET = CuratedTarget(raw_name="variety", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []

