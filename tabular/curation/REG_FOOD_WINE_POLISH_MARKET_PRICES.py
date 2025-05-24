from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: skamlo/wine-price-on-polish-market/wine.csv
====
Examples: 2247
====
URL: https://www.kaggle.com/skamlo/wine-price-on-polish-market/wine.csv
====
Description: 
Wine Price on Polish Market

About Dataset
Context
The data comes from the website wina.pl and was downloaded on September 21, 2023. This dataset can be used for regression tasks. The dataset contains 19 columns:

Content	
name	name of the offer displayed on the website (may contain Polish words or Polish diacritical marks)
price	price of wine (in PLN)
country	the country where the wine was produced
region	the region where the wine was produced
appellation	cocategorization of wine by geo-political boundarie
vineyard	the vineyard where the wine was produced
vintage	production year
volume	bottle volume (in liters)
alcohol	alcohol content in wine (in percentage)
serving temperature	temperature or temperature range at which wine should be served (in degrees Celsius)
color	color of wine
kind	type of wine (champagne, other sparkling wine, sherry or port)
taste	wine taste (dry, semi-dry, sweet or semi-sweet)
style	the lightness of the wine
medals	medals won by wine (individual medals are separated by a comma)
vegans	is wine vegan
natural	is the wine is natural
punctation	average score given by sommeliers (only some wines were rated)
grapes	the grapes from which the wine was produced (individual grapes are separated by a comma)

====
Features:

name (object, 2236 distinct): ['Chateau Musar', "Feudo Arancio Nero d'Avola Sicilia DOC", 'Szampan Ruinart Rose', 'Szampan Rose de Saignee 1er Cru Rene Geoffroy', 'Porto Burmester Colheita', 'Brunello di Montalcino Riserva Capanna', 'Feudo Arancio Grillo Sicilia DOC', 'Paco & Lola Albariño Rías Baixas DO', 'Brut Rose Billecart-Salmon', 'Montelliana Prosecco Spumante Extra Dry Treviso DOC']
price (PLN) (float64, 420 distinct): ['89.0', '99.0', '79.0', '69.0', '59.0', '49.0', '119.0', '109.0', '55.0', '139.0']
country (object, 25 distinct): ['Italy', 'France', 'Spain', 'Austria', 'Poland', 'Portugal', 'Germany', 'Chile', 'Argentina', 'United States of America']
region (object, 136 distinct): ['Tuscany', 'Burgundy', 'Veneto', 'Champagne', 'Trentino Alto Adige', 'Sicily', 'Puglia', 'Rhône Valley', 'Catalonia', 'Piedmont']
appellation (object, 342 distinct): ['Champagne AOC', 'Rioja DOC', 'Toscana IGT', 'Vin de France', 'Alsace AOC', 'Ribera del Duero DO', 'Kamptal DAC', 'Terre Siciliane IGT', 'Venezia Gulia IGT', 'Porto DOC']
vineyard (object, 281 distinct): ['St. Michael-Eppan', 'Cantine San Marzano', 'Maison Michel Chapoutier', 'Auer', 'CAVIT Cantina Viticoltori del Trentino', 'Allegrini', 'Dr. Loosen', 'Niepoort', 'Aresti Chile Wines Limitada', 'Maison Louis Jadot']
vintage (float64, 22 distinct): ['2021.0', '2020.0', '2019.0', '2018.0', '2022.0', '2017.0', '2016.0', '2015.0', '2014.0', '2013.0']
volume (liters) (float64, 11 distinct): ['0.75', '1.5', '0.375', '0.5', '1.0', '3.0', '5.0', '0.7', '0.2', '0.735']
alcohol (%) (float64, 39 distinct): ['13.5', '13.0', '12.5', '14.0', '12.0', '14.5', '11.5', '11.0', '15.0', '10.5']
serving temperature (C) (object, 22 distinct): ['16-18', '16', '10-12', '15', '12', '18', '10', '8', '8-10', '14']
color (object, 4 distinct): ['red', 'white', 'rose', 'orange']
kind (object, 4 distinct): ['sparkling', 'champagne', 'port', 'sherry']
taste (object, 4 distinct): ['dry', 'semi-dry', 'semi-sweet', 'sweet']
style (object, 3 distinct): ['average', 'full', 'light']
medals (object, 45 distinct): ['Mundus Vini Gold', 'Decanter Gold', 'Tre Bicchieri Gambero Rosso', 'Due Bicchieri Gambero Rosso', 'Decanter Silver', 'Decanter Bronze', 'Mundus Vini Silver', 'Berliner Gold', 'IWC Silver', 'IWC Gold']
wegan (bool, 2 distinct): ['0', '1']
natural (bool, 2 distinct): ['0', '1']
punctation (float64, 26 distinct): ['90.0', '93.0', '92.0', '91.0', '94.0', '95.0', '96.0', '91.5', '92.5', '93.5']
grapes (object, 649 distinct): ['Chardonnay', 'Pinot Noir', 'Riesling', 'Tempranillo', 'Sauvignon Blanc', 'Cabernet Sauvignon', 'Sangiovese', 'Primitivo', 'Gewürztraminer', 'Malbec']
'''

CONTEXT = "Information about wines on the polish market"
TARGET = CuratedTarget(raw_name="price (PLN)", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []

DESCRIPTION = '''
Wine Price on Polish Market

About Dataset
Context
The data comes from the website wina.pl and was downloaded on September 21, 2023. This dataset can be used for regression tasks. The dataset contains 19 columns:

Content	
name	name of the offer displayed on the website (may contain Polish words or Polish diacritical marks)
price	price of wine (in PLN)
country	the country where the wine was produced
region	the region where the wine was produced
appellation	cocategorization of wine by geo-political boundarie
vineyard	the vineyard where the wine was produced
vintage	production year
volume	bottle volume (in liters)
alcohol	alcohol content in wine (in percentage)
serving temperature	temperature or temperature range at which wine should be served (in degrees Celsius)
color	color of wine
kind	type of wine (champagne, other sparkling wine, sherry or port)
taste	wine taste (dry, semi-dry, sweet or semi-sweet)
style	the lightness of the wine
medals	medals won by wine (individual medals are separated by a comma)
vegans	is wine vegan
natural	is the wine is natural
punctation	average score given by sommeliers (only some wines were rated)
grapes	the grapes from which the wine was produced (individual grapes are separated by a comma)
'''