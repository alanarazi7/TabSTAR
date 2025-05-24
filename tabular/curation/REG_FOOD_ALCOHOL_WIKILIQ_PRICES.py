from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType
from tabular.preprocessing.textual import remove_currency, remove_percentage

'''
Dataset Name: limtis/wikiliq-dataset/spirits_data.csv
====
Examples: 12869
====
URL: https://www.kaggle.com/limtis/wikiliq-dataset/spirits_data.csv
====
Description: 
Wikiliq - Alcohol dataset (May, 2022)
Liquors, Whiskey, Vodka, Wine, Beer, etc.

About Dataset
Parsed data from WikiliQ website.
Data is presented as-is. It wasn't cleaned, nor modified in any way.

There are three CSV files, containing information about wine, beer and liquors (Including whiskey, beer, vodka, liquors and spirits)

Common columns between these 3 files:
Name (of the product),
Brand,
Country,
Categories (Type of beer/wine/liquor)
ABV (Alcohol by volume)
Suggested Serving Temperature (in Fahrenheit)
Rating (Scale 0 to 5)
Rate count (Amount of reviews, the rating is based on)
Price (Sometimes might be incorrect, be careful)
Volume (Volume, the price is set for)
Description [of a product] (Mostly descriptions, but sometimes may be filled with contacts of a brand, or company's history)

There are additional columns in every file, but names speak for themselves.

====
Features:

Unnamed: 0 (int64, 12869 distinct): ['0', '8584', '8574', '8575', '8576', '8577', '8578', '8579', '8580', '8581']
Name (object, 12804 distinct): ['Hiram Walker Brandy Apricot', 'Leroux Polish Blackberry Flavored Brandy', 'Toschi Lemoncello', 'Tippy Cow Chocolate Rum', 'Whitmeyer’s Texas Peach Whiskey', 'Tippy Cow Vanilla Soft Serve Rum', 'Belvoir Organic Elderflower Presse', 'Montezuma Blue Tequila', 'Margaritaville Coconut Tequila', 'Hiram Walker Kirschwasser']
Country (object, 105 distinct): ['United States', 'Mexico', 'Scotland', 'France', 'Canada', 'United Kingdom', 'Italy', 'Ireland', 'Netherlands', 'Japan']
Brand (object, 3971 distinct): ['DeKuyper Liqueur', 'Bacardi Rum', 'Smirnoff Vodka', 'Jack Daniels', 'Johnnie Walker', 'Gordon  MacPhail', 'Jose Cuervo', 'Pinnacle', 'Absolut', 'Highland Park']
Categories (object, 152 distinct): ['Scotch Whisky, Whiskey', 'Vodka', 'Bourbon, Whiskey', 'Liqueur', 'Ready-to-Drink', 'Rye Whiskey, Whiskey', 'Mezcal', 'American Whiskey, Whiskey', 'Rum', 'Brandy, Cognac']
Tasting Notes (object, 1623 distinct): ['Citrus', 'Smooth', 'Grapefruit', 'Fruity', 'Apple', 'Peach', 'Crisp, Smooth', 'Balanced, Dried Fruit, Rich, Round, Smoky, Smooth, Vanilla', 'Tropical Fruit', 'Mint']
ABV (object, 462 distinct): ['40%', '35%', '45%', '43%', '46%', '30%', '50%', '15%', '42%', '20%']
Base Ingredient (object, 48 distinct): ['Corn', 'Barley', 'Desert Plant', 'Fruit', 'Barley, Corn, Rye', 'Sugar Cane', 'Sugar', 'Wheat', 'Rye', 'Potato']
Years Aged (float64, 49 distinct): ['12.0', '10.0', '15.0', '5.0', '3.0', '2.0', '18.0', '4.0', '8.0', '21.0']
Rating (float64, 17 distinct): ['5.0', '4.8', '4.9', '2.5', '4.7', '4.6', '4.5', '4.4', '3.3', '4.1']
Rate Count (int64, 189 distinct): ['1', '2', '3', '4', '5', '6', '7', '10', '8', '9']
Price (object, 2484 distinct): ['$29.99', '$19.99', '$0.00', '$39.99', '$21.99', '$24.99', '$49.99', '$32.99', '$34.99', '$12.99']
Volume (object, 2 distinct): ['750ml', '1L']
Description (object, 9696 distinct): ['Burnett’s Flavored Vodkas combine the quality of Burnett’s Vodka with all-natural flavors to deliver a superior taste.', '15 pack of 25 oz. prefilled shots', 'SKYY vodka is the first quadruple distilled, triple filtered premium American vodka created in San Francisco in 1992. \xa0SKYY has redefined vodka with flavor by carefully making vodka infused with real fruit.\xa0', 'BACARDÍ Ready to Serve Cocktails are convenient and ready to party. No need to mix, prepare or buy other ingredients, giving you more time to enjoy the moment. Made with BACARDÍ Rum, natural flavors, real ingredients (cane sugar, natural juices or concentrates), and no artificial sweeteners. Consistently high quality and always refreshing, fun and full-flavored at 12.5% ABV.', 'Matured for ten years in American oak ex-bourbon and ex-sherry casks. This award-winning Single Malt was born of Speyside’s natural elements, embodying the rich, fresh sweetness of the region.\n\nGOLD MEDAL | New York International Spirits Competition 2020\nGOLD MEDAL | San Francisco World Spirits Competition 2020\nGOLD MEDAL | San Francisco World Spirits Competition 2019', 'Shottys Party Ready Pre-Packaged Gelatin Shots. Made with Premium Alcohol, natural ingredients, natural flavors, nothing artificial, gluten free, at 12.5% abv, in a fun squeezable cup. Made in the USA!', 'Created in the early-twentieth century using secret family recipes from Europe, Arrow is the #3 best-selling cordial line in America. The product line includes a full array of flavored schnapps, sweet and sour Smackers and brandies, vodka, rum, as well as traditional cordials, liqueurs and creams.', 'Fireball Cinnamon Whisky needs no introduction.  Just imagine what it feels like to stand face-to-face with a fire-breathing dragon who just ate a whisky barrel full of spicy cinnamon. Tastes like heaven, burns like hell. What happens next is up to you.\n\nWhat you have here is smooth whisky with a fiery kick of red hot cinnamon.  Fireball Whisky is best enjoyed straight up as an ice-cold shot. For another way to ignite your nite, try it with your favorite mixer – cola, ginger beer, hard cider, cranberry, etc.  Fireball makes anything taste good!', 'Host the perfect party by serving bar quality Smirnoff Cocktails at home. You are guaranteed the perfect cocktail in every pour.', 'No carbs. No sugar. No sweetener. No preservatives. No additives.\r\n\r\nNütrl Vodka Soda is a wildly light, easy drinking, totally refreshing, sugar free, additive free, 5% alcohol, ready to drink, premium craft vodka soda. It’s the perfect change up from the usual sugary-sweet cooler or high calorie, super-filling beer.']
'''

CONTEXT = "WikiliQ - Alcohol dataset (May, 2022)"
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION,
                       processing_func=remove_currency)
COLS_TO_DROP = ["Unnamed: 0"]
FEATURES = [CuratedFeature(raw_name="ABV", feat_type=FeatureType.NUMERIC,
                           processing_func=remove_percentage),]

DESCRIPTION = '''
Wikiliq - Alcohol dataset (May, 2022)
Liquors, Whiskey, Vodka, Wine, Beer, etc.

About Dataset
Parsed data from WikiliQ website.
Data is presented as-is. It wasn't cleaned, nor modified in any way.

There are three CSV files, containing information about wine, beer and liquors (Including whiskey, beer, vodka, liquors and spirits)

Common columns between these 3 files:
Name (of the product),
Brand,
Country,
Categories (Type of beer/wine/liquor)
ABV (Alcohol by volume)
Suggested Serving Temperature (in Fahrenheit)
Rating (Scale 0 to 5)
Rate count (Amount of reviews, the rating is based on)
Price (Sometimes might be incorrect, be careful)
Volume (Volume, the price is set for)
Description [of a product] (Mostly descriptions, but sometimes may be filled with contacts of a brand, or company's history)

There are additional columns in every file, but names speak for themselves.
'''