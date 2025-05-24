from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: joshuakalobbowles/vivino-wine-data/vivino.csv
====
Examples: 8650
====
URL: https://www.kaggle.com/joshuakalobbowles/vivino-wine-data/vivino.csv
====
Description: 
Vivino Wine Data
Spanish Wine Analysis from Vivino Web Scrape

About Dataset
Context
Wine in Spain is incredibly easy considering price and value. Here is a list of some of Spain's best value wines (https://joshua-k-bowles.medium.com/vivino-insights-into-spanish-wine-1f6e7b78f91f)

Content
Here is a quick summary of value wines from Spain featuring a brief analysis of value based on the Price and Ratings.

Acknowledgements
Data source: (Vivino's Website)

Inspiration
I began this project as I have a personal deep interest in wine and wine regions. My love for nice vino tinto is one of the reasons I choose to live in Spain. This analysis is part of a series I am working on as a portfolio for Data Analysis.

Limitations
The list of wines scraped are not a complete and comprehensive view of Spanish wine, as wine culture in Spain has traditionally been that of small production and barrel sales. For this reason, many small wine producers are not included in Vivinos database and therefore not represented in the data.

The wines shown only include wines currently available for purchase.

As a consumer database, the results and ratings may not represent professional opinions and may be limited by the 5-star rating system due to its overly simplistic nature.

====
Features:

Winery (object, 848 distinct): ['Luzon', 'Tridente', 'Bodegas Olarra', 'Beronia', 'Celler de Capçanes', 'Federico Paternina', 'Bodegas Mano a Mano', 'Valderiz', 'Lavia', 'Baltasar Gracián']
Year (object, 14 distinct): ['2018', '2016', '2015', '2017', '2014', '2013', '2019', '2020', '2012', '2011']
Wine ID (int64, 1527 distinct): ['1230455', '1164695', '2185839', '1175426', '1784470', '1134525', '6773233', '1221284', '1148433', '1515739']
Wine (object, 1662 distinct): ['Crianza 2016', 'Reserva 2016', 'Reserva 2014', 'Rioja Reserva 2014', 'Ribera del Duero Crianza 2018', 'Gran Reserva 2013', 'Reserva Ribera del Duero 2016', 'Triton Tinta de Toro 2018', 'Abadia de Gomariz 2015', 'Altos de Luzón 2016']
Rating (float64, 19 distinct): ['3.8', '4.0', '3.9', '3.7', '4.1', '3.5', '3.6', '4.2', '4.4', '3.4']
num_review (int64, 618 distinct): ['29', '46', '35', '48', '59', '88', '113', '86', '80', '115']
price (float64, 465 distinct): ['12.9', '8.95', '9.95', '9.9', '12.5', '10.9', '7.9', '7.95', '10.5', '6.9']
Country (object, 1 distinct): ['España']
Region (object, 78 distinct): ['Ribera del Duero', 'Rioja', 'Jumilla', 'Calatayud', 'Montsant', 'Toro', 'Empordà', 'Cariñena', 'Castilla', 'Valencia']
'''

CONTEXT = "Vivino Spanish Wine Data"
TARGET = CuratedTarget(raw_name="Rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Wine ID"]
FEATURES = []

DESCRIPTION = '''
Vivino Wine Data
Spanish Wine Analysis from Vivino Web Scrape

About Dataset
Context
Wine in Spain is incredibly easy considering price and value. Here is a list of some of Spain's best value wines (https://joshua-k-bowles.medium.com/vivino-insights-into-spanish-wine-1f6e7b78f91f)

Content
Here is a quick summary of value wines from Spain featuring a brief analysis of value based on the Price and Ratings.

Acknowledgements
Data source: (Vivino's Website)

Inspiration
I began this project as I have a personal deep interest in wine and wine regions. My love for nice vino tinto is one of the reasons I choose to live in Spain. This analysis is part of a series I am working on as a portfolio for Data Analysis.

Limitations
The list of wines scraped are not a complete and comprehensive view of Spanish wine, as wine culture in Spain has traditionally been that of small production and barrel sales. For this reason, many small wine producers are not included in Vivinos database and therefore not represented in the data.

The wines shown only include wines currently available for purchase.

As a consumer database, the results and ratings may not represent professional opinions and may be limited by the 5-star rating system due to its overly simplistic nature.
'''