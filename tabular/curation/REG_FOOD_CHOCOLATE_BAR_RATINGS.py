from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType
from tabular.preprocessing.textual import remove_percentage

'''
Dataset Name: rtatman/chocolate-bar-ratings/flavors_of_cacao.csv
====
Examples: 1795
====
URL: https://www.kaggle.com/rtatman/chocolate-bar-ratings/flavors_of_cacao.csv
====
Description: 
Chocolate Bar Ratings
Expert ratings of over 1,700 chocolate bars

About Dataset
Context
Chocolate is one of the most popular candies in the world. Each year, residents of the United States collectively eat more than 2.8 billions pounds. However, not all chocolate bars are created equal! This dataset contains expert ratings of over 1,700 individual chocolate bars, along with information on their regional origin, percentage of cocoa, the variety of chocolate bean used and where the beans were grown

Flavors of Cacao Rating System:
5= Elite (Transcending beyond the ordinary limits)
4= Premium (Superior flavor development, character and style)
3= Satisfactory(3.0) to praiseworthy(3.75) (well made with special qualities)
2= Disappointing (Passable but contains at least one significant flaw)
1= Unpleasant (mostly unpalatable)

Each chocolate is evaluated from a combination of both objective qualities and subjective interpretation. A rating here only represents an experience with one bar from one batch. Batch numbers, vintages and review dates are included in the database when known.

The database is narrowly focused on plain dark chocolate with an aim of appreciating the flavors of the cacao when made into chocolate. The ratings do not reflect health benefits, social missions, or organic status.

Flavor is the most important component of the Flavors of Cacao ratings. Diversity, balance, intensity and purity of flavors are all considered. It is possible for a straight forward single note chocolate to rate as high as a complex flavor profile that changes throughout. Genetics, terroir, post harvest techniques, processing and storage can all be discussed when considering the flavor component.

Texture has a great impact on the overall experience and it is also possible for texture related issues to impact flavor. It is a good way to evaluate the makers vision, attention to detail and level of proficiency.

Aftermelt is the experience after the chocolate has melted. Higher quality chocolate will linger and be long lasting and enjoyable. Since the aftermelt is the last impression you get from the chocolate, it receives equal importance in the overall rating.

Overall Opinion is really where the ratings reflect a subjective opinion. Ideally it is my evaluation of whether or not the components above worked together and an opinion on the flavor development, character and style. It is also here where each chocolate can usually be summarized by the most prominent impressions that you would remember about each chocolate.

Acknowledgements
These ratings were compiled by Brady Brelinski, Founding Member of the Manhattan Chocolate Society. For up-to-date information, as well as additional content (including interviews with craft chocolate makers), please see his website: Flavors of Cacao

Inspiration
Where are the best cocoa beans grown?
Which countries produce the highest-rated bars?
What’s the relationship between cocoa solids percentage and rating?

====
Features:

Company 
(Maker-if known) (object, 416 distinct): ['Soma', 'Bonnat', 'Fresco', 'Pralus', 'A. Morin', 'Arete', 'Guittard', 'Domori', 'Valrhona', 'Hotel Chocolat (Coppeneur)']
Specific Bean Origin
or Bar Name (object, 1039 distinct): ['Madagascar', 'Peru', 'Ecuador', 'Dominican Republic', 'Venezuela', 'Chuao', 'Sambirano', 'Ocumare', 'Ghana', 'Papua New Guinea']
REF (int64, 440 distinct): ['414', '404', '24', '387', '32', '1454', '1466', '1450', '431', '552']
Review
Date (int64, 12 distinct): ['2015', '2014', '2016', '2012', '2013', '2011', '2009', '2010', '2008', '2007']
Cocoa
Percent (object, 45 distinct): ['70%', '75%', '72%', '65%', '80%', '74%', '68%', '60%', '73%', '85%']
Company
Location (object, 60 distinct): ['U.S.A.', 'France', 'Canada', 'U.K.', 'Italy', 'Ecuador', 'Australia', 'Belgium', 'Switzerland', 'Germany']
Rating (float64, 13 distinct): ['3.5', '3.0', '3.25', '2.75', '3.75', '2.5', '4.0', '2.0', '2.25', '1.5']
Bean
Type (object, 41 distinct): ['\xa0', 'Trinitario', 'Criollo', 'Forastero', 'Forastero (Nacional)', 'Blend', 'Criollo, Trinitario', 'Forastero (Arriba)', 'Criollo (Porcelana)', 'Trinitario, Criollo']
Broad Bean
Origin (object, 100 distinct): ['Venezuela', 'Ecuador', 'Peru', 'Madagascar', 'Dominican Republic', '\xa0', 'Nicaragua', 'Brazil', 'Bolivia', 'Belize']
'''

CONTEXT = "Chocolate Bar expert ratings"
TARGET = CuratedTarget(raw_name="Rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name='Cocoa\nPercent', processing_func=remove_percentage,
                           feat_type=FeatureType.NUMERIC),]

DESCRIPTION = '''
Chocolate Bar Ratings
Expert ratings of over 1,700 chocolate bars

About Dataset
Context
Chocolate is one of the most popular candies in the world. Each year, residents of the United States collectively eat more than 2.8 billions pounds. However, not all chocolate bars are created equal! This dataset contains expert ratings of over 1,700 individual chocolate bars, along with information on their regional origin, percentage of cocoa, the variety of chocolate bean used and where the beans were grown

Flavors of Cacao Rating System:
5= Elite (Transcending beyond the ordinary limits)
4= Premium (Superior flavor development, character and style)
3= Satisfactory(3.0) to praiseworthy(3.75) (well made with special qualities)
2= Disappointing (Passable but contains at least one significant flaw)
1= Unpleasant (mostly unpalatable)

Each chocolate is evaluated from a combination of both objective qualities and subjective interpretation. A rating here only represents an experience with one bar from one batch. Batch numbers, vintages and review dates are included in the database when known.

The database is narrowly focused on plain dark chocolate with an aim of appreciating the flavors of the cacao when made into chocolate. The ratings do not reflect health benefits, social missions, or organic status.

Flavor is the most important component of the Flavors of Cacao ratings. Diversity, balance, intensity and purity of flavors are all considered. It is possible for a straight forward single note chocolate to rate as high as a complex flavor profile that changes throughout. Genetics, terroir, post harvest techniques, processing and storage can all be discussed when considering the flavor component.

Texture has a great impact on the overall experience and it is also possible for texture related issues to impact flavor. It is a good way to evaluate the makers vision, attention to detail and level of proficiency.

Aftermelt is the experience after the chocolate has melted. Higher quality chocolate will linger and be long lasting and enjoyable. Since the aftermelt is the last impression you get from the chocolate, it receives equal importance in the overall rating.

Overall Opinion is really where the ratings reflect a subjective opinion. Ideally it is my evaluation of whether or not the components above worked together and an opinion on the flavor development, character and style. It is also here where each chocolate can usually be summarized by the most prominent impressions that you would remember about each chocolate.

Acknowledgements
These ratings were compiled by Brady Brelinski, Founding Member of the Manhattan Chocolate Society. For up-to-date information, as well as additional content (including interviews with craft chocolate makers), please see his website: Flavors of Cacao

Inspiration
Where are the best cocoa beans grown?
Which countries produce the highest-rated bars?
What’s the relationship between cocoa solids percentage and rating?
'''
