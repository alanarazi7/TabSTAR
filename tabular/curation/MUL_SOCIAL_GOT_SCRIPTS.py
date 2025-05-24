from typing import Optional

from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: albenft/game-of-thrones-script-all-seasons/Game_of_Thrones_Script.csv
====
Examples: 23911
====
URL: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons/Game_of_Thrones_Script.csv
====
Description: 
Game of Thrones Script All Seasons
Script from Every Season of Game of Thrones Extracted from Genius.com

About Dataset
Context
Dataset is generated through a long and complex process. Starting from scrapping the whole URLs provided on Genius.com for Game of Thrones series. Process on scrapping and cleaning the dataset required a lot of time and effort in which I managed to utilize wide range of package available for collecting and compiling data scattered all over the internet.

This dataset is inspired by previous similar dataset published by Ander Fernández Jauregui on https://www.kaggle.com/anderfj/game-of-thrones-series-scripts-breakdowns. I was waiting for him to update the dataset to do some analysis on them. Unfortunately, it was a long time since he last updated the dataset. Therefore, following some of his practice I generated this dataset, and hopefully will be a good use for anyone or at least for my personal analysis.

Content
The content inside is a complete set of Game of Thrones script for all seasons in form of a table containing 6 columns with different data types used for various purposes. Description on each columns are provided on the data description part.

Acknowledgements
Great credits for Genius.com to published the whole script of Game of Thrones series completely. Also, kudos to all of the open source packages out there, as well as people who are contributing on them so we can utilize those packages as we pleases.

Inspiration
There is only one question that I want to find answer using this dataset. Who is the true hero/heroin in the whole series?

====
Features:

Release Date (object, 73 distinct): ['2017-08-13', '2013-04-07', '2014-04-06', '2013-04-28', '2012-04-08', '2011-05-15', '2012-05-06', '2012-04-29', '2012-05-13', '2013-05-12']
Season (object, 8 distinct): ['Season 2', 'Season 3', 'Season 4', 'Season 1', 'Season 5', 'Season 6', 'Season 7', 'Season 8']
Episode (object, 10 distinct): ['Episode 5', 'Episode 2', 'Episode 3', 'Episode 1', 'Episode 7', 'Episode 6', 'Episode 4', 'Episode 8', 'Episode 10', 'Episode 9']
Episode Title (object, 73 distinct): ['Eastwatch', 'Dark Wings, Dark Words', 'Two Swords', 'Kissed by Fire', 'The Night Lands', 'The Wolf and the Lion', 'The Old Gods and the New', 'The Ghost of Harrenhal', 'A Man Without Honor', 'The Bear and the Maiden Fair']
Name (object, 564 distinct): ['tyrion lannister', 'jon snow', 'daenerys targaryen', 'cersei lannister', 'jaime lannister', 'sansa stark', 'arya stark', 'davos', 'theon greyjoy', 'petyr baelish']
Sentence (object, 22299 distinct): ['No.', 'Your Grace.', 'Why?', 'What?', 'Yes.', 'Thank you.', 'Who are you?', 'What are you doing?', 'And?', 'What is it?']
'''

def map_got_character(c: str) -> Optional[str]:
    characters = {'tyrion lannister', 'jon snow', 'daenerys targaryen', 'cersei lannister', 'jaime lannister',
                  'sansa stark', 'arya stark', 'davos', 'theon greyjoy', 'petyr baelish', 'sam', 'bran stark',
                  'bronn', 'jorah mormont', 'tywin lannister', 'varys', 'brienne', 'eddard stark', 'robb stark',
                  'stannis baratheon', 'catelyn stark', 'ramsay bolton', 'margaery tyrell', 'joffrey lannister',
                  'melisandre', 'sandor clegane', 'shae', 'gendry baratheon', 'tormund', 'gilly', 'missandei',
                  'olenna tyrell', 'ygritte', 'daario', 'sam tarly', 'podrick', 'sparrow', 'yara greyjoy',
                  'osha', 'oberyn martell', 'tommen lannister', 'robert baratheon', 'grey worm'}
    if c in characters:
        return c
    return None

CONTEXT = "Game of Thrones Scripts All Seasons"
TARGET = CuratedTarget(raw_name='Name', new_name="Character", task_type=SupervisedTask.MULTICLASS,
                       processing_func=map_got_character)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Release Date", feat_type=FeatureType.DATE)]

DESCRIPTION = '''
Game of Thrones Script All Seasons
Script from Every Season of Game of Thrones Extracted from Genius.com

About Dataset
Context
Dataset is generated through a long and complex process. Starting from scrapping the whole URLs provided on Genius.com for Game of Thrones series. Process on scrapping and cleaning the dataset required a lot of time and effort in which I managed to utilize wide range of package available for collecting and compiling data scattered all over the internet.

This dataset is inspired by previous similar dataset published by Ander Fernández Jauregui on https://www.kaggle.com/anderfj/game-of-thrones-series-scripts-breakdowns. I was waiting for him to update the dataset to do some analysis on them. Unfortunately, it was a long time since he last updated the dataset. Therefore, following some of his practice I generated this dataset, and hopefully will be a good use for anyone or at least for my personal analysis.

Content
The content inside is a complete set of Game of Thrones script for all seasons in form of a table containing 6 columns with different data types used for various purposes. Description on each columns are provided on the data description part.

Acknowledgements
Great credits for Genius.com to published the whole script of Game of Thrones series completely. Also, kudos to all of the open source packages out there, as well as people who are contributing on them so we can utilize those packages as we pleases.

Inspiration
There is only one question that I want to find answer using this dataset. Who is the true hero/heroin in the whole series?
'''