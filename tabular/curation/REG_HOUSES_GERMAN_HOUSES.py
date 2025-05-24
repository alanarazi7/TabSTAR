from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: German-House-Prices
====
URL: https://www.openml.org/search?type=data&id=43342
====
Description: Context
Projects are a great way to learn data science. So I started my own. The numerous housing data sets on Kaggle were the inspiration for this data set. Predicting housing prices is a simple yet insightful regression problem. Understanding data takes time, and the more people analyze it, the faster the secrets can be uncovered. I acquired the data by scraping Immo Scout 24, a marketplace for German real estate.
====
Features:

Unnamed:_0 (numeric, 10552 distinct): ['0', '7028', '7030', '7031', '7032', '7033', '7034', '7035', '7036', '7037']
Price (numeric, 1411 distinct): ['399000.0', '349000.0', '299000.0', '249000.0', '450000.0', '449000.0', '350000.0', '395000.0', '499000.0', '549000.0']
Type (string, 12 distinct): ['Mid-terrace house', 'Duplex', 'Single dwelling', 'Farmhouse', 'Villa', 'Multiple dwelling', 'Residential property', 'Special property', 'Bungalow', 'Corner house']
Living_space (numeric, 1867 distinct): ['120.0', '150.0', '200.0', '160.0', '130.0', '140.0', '180.0', '100.0', '170.0', '110.0']
Lot (numeric, 2526 distinct): ['1000.0', '800.0', '500.0', '600.0', '300.0', '400.0', '700.0', '200.0', '1200.0', '250.0']
Usable_area (numeric, 5996 distinct): ['50.0', '100.0', '60.0', '30.0', '40.0', '80.0', '70.0', '20.0', '120.0', '90.0']
Free_of_Relation (string, 705 distinct): [' nach Absprache ', ' sofort ', ' nach Vereinbarung ', ' ab sofort ', ' Nach Vereinbarung ', ' 01.08.2020 ', ' 01.10.2020 ', ' 01.09.2020 ', ' Sofort ', ' Nach Absprache ']
Rooms (numeric, 72 distinct): ['5.0', '6.0', '4.0', '7.0', '8.0', '9.0', '10.0', '3.0', '12.0', '11.0']
Bedrooms (numeric, 3706 distinct): ['3.0', '4.0', '5.0', '2.0', '6.0', '1.0', '7.0', '8.0', '9.0', '10.0']
Bathrooms (numeric, 1829 distinct): ['2.0', '1.0', '3.0', '4.0', '5.0', '6.0', '8.0', '7.0', '10.0', '9.0']
Floors (numeric, 2674 distinct): ['2.0', '3.0', '1.0', '4.0', '5.0', '0.0', '6.0', '7.0', '8.0', '13.0']
Year_built (numeric, 986 distinct): ['1900.0', '1960.0', '2020.0', '2000.0', '1978.0', '1972.0', '1950.0', '1920.0', '1970.0', '1980.0']
Furnishing_quality (string, 5 distinct): ['normal', 'basic', 'refined', 'luxus']
Year_renovated (numeric, 5270 distinct): ['2019.0', '2018.0', '2017.0', '2020.0', '2015.0', '2016.0', '2010.0', '2014.0', '2012.0', '2013.0']
Condition (string, 11 distinct): ['modernized', 'refurbished', 'dilapidated', 'maintained', 'renovated', 'fixer-upper', 'first occupation after refurbishment', 'first occupation', 'by arrangement', 'as new']
Heating (string, 14 distinct): ['stove heating', 'heat pump', 'central heating', 'oil heating', 'underfloor heating', 'night storage heater', 'district heating', 'wood-pellet heating', 'floor heating', 'electric heating']
Energy_source (string, 105 distinct): [' Gas ', ' l ', ' Strom ', ' Fernwrme ', ' Erdgas leicht ', ' Holzpellets ', ' Erdwrme ', ' Holz ', ' Flssiggas ', ' Solar, Gas ']
Energy_certificate (string, 4 distinct): ['available', 'not required by law', 'available for inspection']
Energy_certificate_type (string, 3 distinct): ['demand certificate', 'consumption certificate']
Energy_consumption (numeric, 9542 distinct): ['114.0', '120.0', '128.0', '121.0', '130.0', '97.0', '78.0', '125.0', '131.0', '119.0']
Energy_efficiency_class (string, 10 distinct): [' D ', ' F ', ' H ', ' E ', ' G ', ' C ', ' B ', ' A+ ', ' A ']
State (string, 17 distinct): ['Nordrhein-Westfalen', 'Bayern', 'Baden-Wrttemberg', 'Niedersachsen', 'Rheinland-Pfalz', 'Hessen', 'Schleswig-Holstein', 'Sachsen', 'Brandenburg', 'Sachsen-Anhalt']
City (string, 535 distinct): ['Hannover (Kreis)', 'Nordfriesland (Kreis)', 'Rhein-Neckar-Kreis', 'Wetteraukreis', 'Rhein-Sieg-Kreis', 'Gifhorn (Kreis)', 'Mayen-Koblenz (Kreis)', 'Schleswig-Flensburg (Kreis)', 'Mittelsachsen (Kreis)', 'Rendsburg-Eckernfrde (Kreis)']
Place (string, 4761 distinct): ['Innenstadt', 'Falkensee', 'Stadtmitte', 'Homburg', 'Hameln', 'Barsinghausen', 'Schotten', 'Celle', 'Gifhorn', 'Cuxhaven']
Garages (numeric, 1997 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '7.0', '10.0', '12.0']
Garagetype (string, 8 distinct): ['Garage', 'Outside parking lot', 'Parking lot', 'Carport', 'Underground parking lot', 'Duplex lot', 'Car park lot']
'''

CONTEXT = "German House Prices"
COLS_TO_DROP = ['Unnamed:_0']
TARGET = CuratedTarget(raw_name='Price', task_type=SupervisedTask.REGRESSION)
FEATURES = []
