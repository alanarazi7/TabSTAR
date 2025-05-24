from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: neilcosgrove/scotch-whiskey-reviews-update-2020/scotch_review2020.csv
====
Examples: 2247
====
URL: https://www.kaggle.com/neilcosgrove/scotch-whiskey-reviews-update-2020/scotch_review2020.csv
====
Description: 
Scotch Whiskey Reviews Update 2020
A listing of 2,247 reviews of Scotch Whiskys

About Dataset
Context
This a compilation of Scotch Whisky Reviews from the "Whisky Advocate" taken as of January 2021 (so results are up to 2020) with 2,247 reviews

Acknowledgements
Full credit for the script goes to the Kaggle user thatdataanalyst and his dataset 2,2k+ Scotch Whisky Reviews on Kaggle. I noticed that the dataset was over 3 years old, and I meerly executed the existing script. I provide the data here for those who may wish to use it, but if you find it useful please give him the upvote.

====
Features:

id (int64, 2247 distinct): ['1', '1502', '1496', '1497', '1498', '1499', '1500', '1501', '1503', '1494']
name (object, 2187 distinct): ['Compass Box The Peat Monster, 46%', 'Longmorn 16 year old, 48%', 'Laphroaig Triple Wood, 48%', 'Bruichladdich Bere Barley 2010, 50%', 'Kilchoman Loch Gorm 2018, 46%', 'Ardbeg An Oa, 46.6%', 'Tomatin 30 year old, 46%', 'Highland Queen Majesty 15 year old, 46%', 'Compass Box Spice Tree Extravaganza, 46%', 'Sheep Dip Islay Blended Malt, 40%']
category (object, 3 distinct): ['Single Malt Scotch', 'Blended Scotch Whisky', 'Blended Malt Scotch Whisky']
review.point (int64, 15 distinct): ['88', '87', '90', '89', '86', '85', '92', '91', '84', '93']
price (object, 474 distinct): ['100', '65', '70', '60', '80', '50', '120', '90', '150', '55']
currency (object, 1 distinct): ['$']
description.1.2247. (object, 2207 distinct): ['Swiss-based Chapter 7 released this 19 year old single malt, a marriage of two sherry butts (#796 and #1,476). Malt, sweet sherry, cocktail cherries, milky coffee, and a slightly earthy undertone on the nose. Very rich on the palate, with chewy-sweet soft fruits, notably strawberries, syrup sponge, and lively fruit spices. Mouth-drying in the long finish, with aniseed and wood spice. (1,076 bottles)', "What impresses me most is how this whisky evolves; it's incredibly complex. On the nose and palate, this is a thick, viscous, whisky with notes of sticky toffee, earthy oak, fig cake, roasted nuts, fallen fruit, pancake batter, black cherry, ripe peach, dark chocolate-covered espresso bean, polished leather, tobacco, a hint of wild game, and lingering, leafy damp kiln smoke. Flavors continue on the palate long after swallowing. This is what we all hope for (and dream of) in an older whisky!", 'The now-annual unpeated release shows its high strength on the nose, but under the burn is a clean, mineral, and slightly lean Caol Ila with just a tiny whiff of smoke. A mix of grassiness/herbal notes, with delicate white fruits that plump out into tinned fruit salad, gooseberry, and fresh pineapple. The palate is sweet and cake-like, while the heat enhances its salty tang. Delightful, sweet, and long. (10,668 bottles)', 'Just eight whiskies in the blend, married and finished in first fill Spanish sherry and bourbon casks. An insistent nose, crackling with spices, with toasted Eccles cake anointed with grated nutmeg, vanilla extract, cassia, and dark soy sauce. Light honey and vanilla, tangerine oils, and lime peel exhibit perfectly-paced development, with flavor building over a minute or more. Warming ginger, spices, and tropical fruits of guava and papaya close out this first annual special edition. Impressive work.', 'Lord Elcho was an 18th century ancestor of William Wemyss, who fought on Bonnie Prince Charlie’s side at the Battle of Culloden in 1745. With a minimum of 40% malt, this fine blend has a rather perfumed nose of fresh mint, green apple, sliced melon, and tropical fruits. The soft candy sugar and butterscotch palate builds, with layers of malt, cherry laces, gingerbread, and pfeffernüsse leading to a ginger and spice finish of significant length. Highly accomplished.\xa0£26', 'The nose is sweet (think barley sugar/boiled sweets) with little bits of wheat chaff flying around in the background with dried flower petals and drying cut grass. Opens dramatically with water into almond milk/horchata and flowers. The palate is sweet and lifted with those gentle florals to the fore. Instead of Tormore’s normal nagging rigidity, this flows sweetly over the tongue, leaving fruit leather, stewed rhubarb, and with water, rosewater and fresh wild strawberry. A lovely Tormore!\xa0£118', 'Golden, lifted, and aromatic. The fleshy ripeness of the 13 year old is still there, but that little sulfur edge has now gone, revealing the ripe fruits massing underneath. Now you find pineapple and light chalk. The flowers have become daffodils and bluebells rather than lily, along with a soft, vanilla ice cream plumpness. Sweet and full, and just a shade lighter than the 13 year old. Muscly, but sweet; that’s the paradox of the Craigellachie character.\xa0£83', 'The latest releases from Glengoyne distillery are 10 year olds, one matured in sherry wood and one in bourbon barrels, as was the case with release 5. 9,000 bottles are available globally. Initially savory on the nose, slightly earthy, with sherry, new leather, lemonade, and a hint of ozone. Spicy and zesty, with developing stewed fruits, dark chocolate, and deep sherry notes on the palate. The finish is long and persistently spicy, mildly smoky, with quite dry sherry notes.', 'Last year’s was a top-notch, defiantly sherried example of Bowmore. This year’s batch thrusts equally boldly, but starts in a more Japanese-accented fashion: think soy, miso paste, and salmon teriyaki. Light leather, with hickory campfire smoke coming through strongly. The big, oily, tarry palate is like a spent barbecue with a hint of skidding car tires on Bowmore High Street. So, still a belter, but why so limited? Beam Suntory, please sort it out! (6,000 bottles)\xa0£60', 'This expression from Aberfeldy was distilled in April 1995 and bottled in January 2014 after maturation in refill, remade sherry hogsheads (casks #2,488, 2,489, and 2,491). The nose is floral, with ginger and developing milk chocolate. Progressively sweeter, with slight sherry and vanilla notes. The palate is silky and sweet, with banoffee pie, peaches, and spicy oak. The finish is long, with cocoa powder and more spiced oak.\xa0£60']
'''

CONTEXT = "Scotch Whiskey Reviews 2020"
TARGET = CuratedTarget(raw_name='review.point', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id"]
FEATURES = [CuratedFeature(raw_name='description.1.2247.', new_name="Review")]

DESCRIPTION = '''
Scotch Whiskey Reviews Update 2020
A listing of 2,247 reviews of Scotch Whiskys

About Dataset
Context
This a compilation of Scotch Whisky Reviews from the "Whisky Advocate" taken as of January 2021 (so results are up to 2020) with 2,247 reviews

Acknowledgements
Full credit for the script goes to the Kaggle user thatdataanalyst and his dataset 2,2k+ Scotch Whisky Reviews on Kaggle. I noticed that the dataset was over 3 years old, and I meerly executed the existing script. I provide the data here for those who may wish to use it, but if you find it useful please give him the upvote.
'''