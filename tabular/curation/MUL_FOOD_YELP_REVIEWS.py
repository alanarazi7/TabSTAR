from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: omkarsabnis/yelp-reviews-dataset/yelp.csv
====
Examples: 10000
====
URL: https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset/yelp.csv
====
Description: 
# https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset 

Yelp Dataset
A trove of reviews, businesses, users, tips, and check-in data!

About Dataset
Context
This dataset is a subset of Yelp's businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In the most recent dataset you'll find information about businesses across 8 metropolitan areas in the USA and Canada.

Content
This dataset contains five JSON files and the user agreement.
More information about those files can be found here.

====
Features:

business_id (object, 4174 distinct): ['JokKtdXU7zXHcr20Lrk29A', 'ntN85eu27C04nwyPa8IHtw', 'hW0Ne_HTHEAgGF1rAdmR-g', 'WNy1uzcmm_UHmTyR--o5IA', 'VVeogjZya58oiTxK7qUjAQ', 'V1nEpIRmEa1768oj_tuxeQ', 'SDwYQ6eSu1htn8vHWv128g', 'uKSX1n1RoAzGq4bV8GPHVg', 'EWMwV5V9BxNs_U6nNVMeqw', '-sC66z4SO3tR7nFCjfQwuQ']
date (object, 1995 distinct): ['2011-03-28', '2012-01-03', '2012-04-12', '2012-11-25', '2012-05-10', '2012-07-12', '2011-12-30', '2012-06-18', '2012-07-10', '2012-08-18']
review_id (object, 10000 distinct): ['fWKvX83p0-ka4JS3dc6E5A', 'FQauZH0pL52wlkquBCxuPQ', 'TRj12tpfNpD_7rV1GEUoiw', 'gqbCVFKy4w9N57fgTn7OFw', 'xtuAHT4aRsAXUhfqV7CoGg', 'xISa9KeJRCRlPTuYmp1qhQ', 'JFJRcKeqNObG_zk_jUJgiw', 'jXM6_jAixq0rj3bOmZNDdQ', 'EFsIM_LA8jOwrlGyFrJ5mA', 'mDKTRuPb4i9rAitkiYLBrA']
stars (int64, 5 distinct): ['4', '5', '3', '2', '1']
text (object, 9998 distinct): ['Great service', "This review is for the chain in general. The location we went to is new so it isn't in Yelp yet. Once it is I will put this review there as well. We were there on Friday at 5 PM. \n\nThe reason I gave it 2 stars is because the burger was very good and it was made the way I asked for it. My husbands burger was not.\n\nBut, the server and the fries left a lot to be desired. Let me preface by saying that we had been to several other locations. I like my fries crispy. I ask for them well done, extra crispy, scorched, tortured hollow tubes. Whatever their buzz word is for well done. The location will comply. EVERY OTHER 5 GUYS HAS COMPLIED. But not the one at TATUM AND SHEA. She said that corporate said they are not to cook the fries that way. So if we were to put up with soggy fries - yes soggy, then we did not want them. \n\nShe also interrupted us several times which is rude. THEN she went and called corporate just to double check for us and she came to the table and said they said no they were not to cook them that way. Seriously? We did not ask for her to do this. She actually accused us of being undercover shoppers. We started to say something and then again- she interupted.\n\nListen, if you explain that our choice is not how the company wishes to present their product and we still choose to have them a different way, you should comply. It is after all our money and our decision. I was raised with the rules that #1 the customer is always right. And #2 if the customer is wrong REFER TO RULE NUMBER 1!!\n\nWe will not return. They have lost our business and I hope she loses her job.\nIf you want to try a really good burger AND FRIES place- go to Paradise Valley Burger Company at 40th Street and Bell. You will not be disappointed.", 'My wife took me here on my birthday for breakfast and it was excellent.  The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning.  It looked like the place fills up pretty quickly so the earlier you get here the better.\n\nDo yourself a favor and get their Bloody Mary.  It was phenomenal and simply the best I\'ve ever had.  I\'m pretty sure they only use ingredients from their garden and blend them fresh when you order it.  It was amazing.\n\nWhile EVERYTHING on the menu looks excellent, I had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  It came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  It was the best "toast" I\'ve ever had.\n\nAnyway, I can\'t wait to go back!', 'Great ambiance, good food.  My husband and I had the Southwest Benedict and Eggs Benedict, both were delicious.  I do wish the eggs were a bit more runny and the food was warmer, but the presentation was beautiful.', 'Do you hate shaving as much as I do? Do you find waxing too painful and expensive every 6-8 weeks?  Well I have the answer for you!!!  Go see Cindy Semerdjian at Cosmetic Laser Solutions.  I was introduced to Cindy through a friend AND NOW I AM HOOKED!!!   Not only do I no longer have hair, it is very inexpensive!  How much did it cost for you to get waxed...what would you say if I told you for a few dollars more the hair would be gone...PERMANENTLY!   \n\nCindy offers a free initial consultation.  A few prices that I have experienced on a per session basis is: underarm - $100 ; bikini (brazilian) - $100; full leg (both) - $250.\n\nI highly, highly recommend going to see Cindy!  Tell her Jillian sent you!    \n\nSo get started today to get ready for summer!', 'This place is awesome they r going to replace a bad  tire for me for free that another discount tire put on they r awesome', "I ordered takeout from here recently...and it was delicious! I phoned in my order during a busy dinner time, so it was a bit chaotic and the food took a longer than they had quoted me, but it was worth the wait. I had the imperial rolls, which are like veggie-filled non-greasy egg rolls, and the pad thai. I wasn't sure about a vegetarian pad thai without chicken/shrimp, but neither was missed because the flavor was amazing. Definitely will be ordering food from here again--just maybe during a less busy time.", "Phoenix has pretty slim pickings when it comes to good pizza, or at least pizza that I'm accustomed to.  Rosati's, while not quite what I'm used to, is the best representation of Chicago style pizza that I've found.  \n\nThe Chicago style deep dish is good.  The toppings, especially the sausage, are plentiful and tasty and the sauce is good.  I think the crust is what is a bit off.  \n\nOverall, Rosati's is still my choice for pizza in the valley.", "The parking lot is too small so watch out for your car doors. \n\nThe seafood is about as authentic as it gets in the US. Arizonans cook their seafood like New Yorkers cook their steaks: rare = medium, and medium = shriveled. Still, the shrimp & octopus cocktail was tender & plump. The peppers stuffed with shrimp were savory. The rice was not sticky and the refried beans didn't taste like they had lard. But I didn't ask.\n\nI would do this again, but next time I'll bring a friend. It's a very social place, 4 girlfriends chatted it up before, during and after I left.", 'This is the pub burger you have been looking for. Nothing fancy, just a good solid burger with a ice cold beer. Be aware this is cash only, but unless you fall in you wont need much.']
type (object, 1 distinct): ['review']
user_id (object, 6403 distinct): ['fczQCSmaWF78toLEmb0Zsw', '0CMz8YaO3f8xu4KqQgKb9Q', '90a6z--_CUrl84aCzZyPsg', '0mqHhdKEdak_A1FBhFNXqA', '4ozupHULqGyO42s3zNUzOQ', 'wHg1YkCzdZq9WBJOTRgxHQ', '_PzSNcfrCjeBxSLXRoMmgQ', '0bNXP9quoJEgyVZu9ipGgQ', 'rLtl8ZkDX5vH5nAx9C3q5Q', 'joIzw_aUiNvBTuGoytrH7g']
cool (int64, 29 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
useful (int64, 28 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
funny (int64, 29 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''

CONTEXT = "YELP Dataset Reviews"
TARGET = CuratedTarget(raw_name="stars", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["user_id", "business_id", "review_id"]
FEATURES = [CuratedFeature(raw_name="date", feat_type=FeatureType.DATE),]

DESCRIPTION = '''
# https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset 

Yelp Dataset
A trove of reviews, businesses, users, tips, and check-in data!

About Dataset
Context
This dataset is a subset of Yelp's businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In the most recent dataset you'll find information about businesses across 8 metropolitan areas in the USA and Canada.

Content
This dataset contains five JSON files and the user agreement.
More information about those files can be found here.
'''
