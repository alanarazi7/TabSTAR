from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: rounakbanik/the-movies-dataset/movies_metadata.csv
====
Examples: 45466
====
URL: https://www.kaggle.com/rounakbanik/the-movies-dataset/movies_metadata.csv
====
Description: 
The Movies Dataset
Metadata on over 45,000 movies. 26 million ratings from over 270,000 users.

About Dataset
Context
These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.

This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

Content
This dataset consists of the following files:

movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.

The Full MovieLens Dataset consisting of 26 million ratings and 750,000 tag applications from 270,000 users on all the 45,000 movies in this dataset can be accessed here

Acknowledgements
This dataset is an ensemble of data collected from TMDB and GroupLens.
The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This product uses the TMDb API but is not endorsed or certified by TMDb. Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. You can try it for yourself here.

The Movie Links and Ratings have been obtained from the Official GroupLens website. The files are a part of the dataset available here

====
Features:

adult (object, 5 distinct): ['False', 'True', ' - Written by Ørnås', ' Rune Balot goes to a casino connected to the October corporation to try to wrap up her case once and for all.', ' Avalanche Sharks tells the story of a bikini contest that turns into a horrifying affair when it is hit by a shark avalanche.']
belongs_to_collection (object, 1698 distinct): ["{'id': 415931, 'name': 'The Bowery Boys', 'poster_path': '/q6sA4bzMT9cK7EEmXYwt7PNrL5h.jpg', 'backdrop_path': '/foe3kuiJmg5AklhtD3skWbaTMf2.jpg'}", "{'id': 421566, 'name': 'Totò Collection', 'poster_path': '/4ayJsjC3djGwU9eCWUokdBWvdLC.jpg', 'backdrop_path': '/jaUuprubvAxXLAY5hUfrNjxccUh.jpg'}", "{'id': 645, 'name': 'James Bond Collection', 'poster_path': '/HORpg5CSkmeQlAolx3bKMrKgfi.jpg', 'backdrop_path': '/6VcVl48kNKvdXOZfJPdarlUGOsk.jpg'}", "{'id': 96887, 'name': 'Zatôichi: The Blind Swordsman', 'poster_path': '/8Q31DAtmFJjhFTwQGXghBUCgWK2.jpg', 'backdrop_path': '/bY8gLImMR5Pr9PaG3ZpobfaAQ8N.jpg'}", "{'id': 37261, 'name': 'The Carry On Collection', 'poster_path': '/2P0HNrYgKDvirV8RCdT1rBSJdbJ.jpg', 'backdrop_path': '/38tF1LJN7ULeZAuAfP7beaPMfcl.jpg'}", "{'id': 34055, 'name': 'Pokémon Collection', 'poster_path': '/j5te0YNZAMXDBnsqTUDKIBEt8iu.jpg', 'backdrop_path': '/iGoYKA0TFfgSoZpG2u5viTJMGfK.jpg'}", "{'id': 413661, 'name': 'Charlie Chan (Sidney Toler) Collection', 'poster_path': '/y0xWQpLRattvypZXF5ZiuipsD2U.jpg', 'backdrop_path': None}", "{'id': 374509, 'name': 'Godzilla (Showa) Collection', 'poster_path': '/scvwS6k8gIW8w24UcmePQqVL10l.jpg', 'backdrop_path': '/dx9YSup5zEOjxYwG4UkYBVAZIXo.jpg'}", "{'id': 425164, 'name': 'Dragon Ball Z (Movie) Collection', 'poster_path': '/2VMZ1zRFPnUQtQp5K4WRXvDYBjh.jpg', 'backdrop_path': '/7PcbijxTfwi9vjWEfXdS0ReAw8q.jpg'}", "{'id': 38451, 'name': 'Charlie Chan (Warner Oland) Collection', 'poster_path': '/eSDdu6pbocmayu1SXQFU9VYYoQ6.jpg', 'backdrop_path': '/9bE62qBanBFtoiIc9cXjk1xW3w.jpg'}"]
budget (object, 1226 distinct): ['0', '5000000', '10000000', '20000000', '2000000', '15000000', '3000000', '25000000', '1000000', '30000000']
genres (object, 4069 distinct): ["[{'id': 18, 'name': 'Drama'}]", "[{'id': 35, 'name': 'Comedy'}]", "[{'id': 99, 'name': 'Documentary'}]", '[]', "[{'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]", "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}]", "[{'id': 27, 'name': 'Horror'}]", "[{'id': 35, 'name': 'Comedy'}, {'id': 10749, 'name': 'Romance'}]", "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]", "[{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}]"]
homepage (object, 7673 distinct): ['http://www.georgecarlin.com', 'http://www.wernerherzog.com/films-by.html', 'http://breakblade.jp/', 'http://www.thehungergames.movie/', 'http://www.crownintlpictures.com/tztitles.html', 'http://www.transformersmovie.com/', 'http://phantasm.com', 'http://www.kungfupanda.com/', 'http://www.crownintlpictures.com/actitles.html', 'http://www.crownintlpictures.com/ostitles.html']
id (object, 45436 distinct): ['141971', '168538', '25541', '15028', '11115', '84198', '13209', '77221', '152795', '12600']
imdb_id (object, 45417 distinct): ['tt1180333', '0', 'tt0270288', 'tt0157472', 'tt0446676', 'tt0499537', 'tt1701210', 'tt1736049', 'tt0067306', 'tt0046468']
original_language (object, 92 distinct): ['en', 'fr', 'it', 'ja', 'de', 'es', 'ru', 'hi', 'ko', 'zh']
original_title (object, 43373 distinct): ['Hamlet', 'Alice in Wonderland', 'Les Misérables', 'Cinderella', 'Macbeth', 'A Christmas Carol', 'The Three Musketeers', 'Wuthering Heights', 'Frankenstein', 'Blackout']
overview (object, 44307 distinct): ['No overview found.', 'No Overview', ' ', 'No movie overview available.', 'A few funny little novels about different aspects of life.', 'Recovering from a nail gun shot to the head and 13 months of coma, doctor Pekka Valinta starts to unravel the mystery of his past, still suffering from total amnesia.', "King Lear, old and tired, divides his kingdom among his daughters, giving great importance to their protestations of love for him. When Cordelia, youngest and most honest, refuses to idly flatter the old man in return for favor, he banishes her and turns for support to his remaining daughters. But Goneril and Regan have no love for him and instead plot to take all his power from him. In a parallel, Lear's loyal courtier Gloucester favors his illegitimate son Edmund after being told lies about his faithful son Edgar. Madness and tragedy befall both ill-starred fathers.", 'Adaptation of the Jane Austen novel.', 'Released', "With friends like these, who needs enemies? That's the question bad guy Porter is left asking after his wife and partner steal his heist money and leave him for dead -- or so they think. Five months and an endless reservoir of bitterness later, Porter's partners and the crooked cops on his tail learn how bad payback can be."]
popularity (object, 44176 distinct): ['0.0', '0.0', '1e-06', '0.0008', '0.0', '0.0006', '0.00022', '0.0003', '0.0012', '0.000308']
poster_path (object, 45024 distinct): ['/5D7UBSEgdyONE6Lql6xS7s6OLcW.jpg', '/qW1oQlOHizRHXZQrpkimYr0oxzn.jpg', '/2kslZXOaW0HmnGuVPCnQlCdXFR9.jpg', '/8VSZ9coCzxOCW2wE2Qene1H1fKO.jpg', '/cdwVC18URfEdQjjxqJyRMoGDC0H.jpg', '/5ILjS6XB5deiHop8SXPsYxXWVPE.jpg', '/k0MF0IIbJ2PfOIku2KyraXL72d8.jpg', '/cYLp3nPDXg1lT7Esebevp6K57tH.jpg', '/gLVRTxaLtUDkfscFKPyYrCtRnTk.jpg', '/jn8L1QdWWX5c0NUOLjzaSXtZrbt.jpg']
production_companies (object, 22708 distinct): ['[]', "[{'name': 'Metro-Goldwyn-Mayer (MGM)', 'id': 8411}]", "[{'name': 'Warner Bros.', 'id': 6194}]", "[{'name': 'Paramount Pictures', 'id': 4}]", "[{'name': 'Twentieth Century Fox Film Corporation', 'id': 306}]", "[{'name': 'Universal Pictures', 'id': 33}]", "[{'name': 'RKO Radio Pictures', 'id': 6}]", "[{'name': 'Columbia Pictures Corporation', 'id': 441}]", "[{'name': 'Columbia Pictures', 'id': 5}]", "[{'name': 'Mosfilm', 'id': 5120}]"]
production_countries (object, 2393 distinct): ["[{'iso_3166_1': 'US', 'name': 'United States of America'}]", '[]', "[{'iso_3166_1': 'GB', 'name': 'United Kingdom'}]", "[{'iso_3166_1': 'FR', 'name': 'France'}]", "[{'iso_3166_1': 'JP', 'name': 'Japan'}]", "[{'iso_3166_1': 'IT', 'name': 'Italy'}]", "[{'iso_3166_1': 'CA', 'name': 'Canada'}]", "[{'iso_3166_1': 'DE', 'name': 'Germany'}]", "[{'iso_3166_1': 'RU', 'name': 'Russia'}]", "[{'iso_3166_1': 'IN', 'name': 'India'}]"]
release_date (object, 17336 distinct): ['2008-01-01', '2009-01-01', '2007-01-01', '2005-01-01', '2006-01-01', '2002-01-01', '2004-01-01', '2001-01-01', '2003-01-01', '1997-01-01']
revenue (float64, 6863 distinct): ['0.0', '12000000.0', '11000000.0', '10000000.0', '2000000.0', '6000000.0', '5000000.0', '500000.0', '8000000.0', '1.0']
runtime (float64, 353 distinct): ['90.0', '0.0', '100.0', '95.0', '93.0', '96.0', '92.0', '94.0', '91.0', '88.0']
spoken_languages (object, 1931 distinct): ["[{'iso_639_1': 'en', 'name': 'English'}]", '[]', "[{'iso_639_1': 'fr', 'name': 'Français'}]", "[{'iso_639_1': 'ja', 'name': '日本語'}]", "[{'iso_639_1': 'it', 'name': 'Italiano'}]", "[{'iso_639_1': 'es', 'name': 'Español'}]", "[{'iso_639_1': 'ru', 'name': 'Pусский'}]", "[{'iso_639_1': 'de', 'name': 'Deutsch'}]", "[{'iso_639_1': 'en', 'name': 'English'}, {'iso_639_1': 'fr', 'name': 'Français'}]", "[{'iso_639_1': 'en', 'name': 'English'}, {'iso_639_1': 'es', 'name': 'Español'}]"]
status (object, 6 distinct): ['Released', 'Rumored', 'Post Production', 'In Production', 'Planned', 'Canceled']
tagline (object, 20283 distinct): ['Based on a true story.', 'Be careful what you wish for.', 'Trust no one.', '-', 'Classic Albums', 'Which one is the first to return - memory or the murderer?', 'Who is John Galt?', 'Some doors should never be opened.', 'There is no turning back', 'Documentary']
title (object, 42277 distinct): ['Cinderella', 'Hamlet', 'Alice in Wonderland', 'Beauty and the Beast', 'Les Misérables', 'The Three Musketeers', 'Blackout', 'A Christmas Carol', 'Treasure Island', 'King Lear']
video (object, 2 distinct): ['0', '1']
vote_average (float64, 92 distinct): ['0.0', '6.0', '5.0', '7.0', '6.5', '6.3', '5.5', '5.8', '6.4', '6.7']
vote_count (float64, 1820 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
'''

CONTEXT = "Metadata of movies released until 2017 for box-office revenues"
TARGET = CuratedTarget(raw_name="revenue", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id", "imdb_id", "poster_path",]
FEATURES = []

DESCRIPTION = '''
The Movies Dataset
Metadata on over 45,000 movies. 26 million ratings from over 270,000 users.

About Dataset
Context
These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.

This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

Content
This dataset consists of the following files:

movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.

The Full MovieLens Dataset consisting of 26 million ratings and 750,000 tag applications from 270,000 users on all the 45,000 movies in this dataset can be accessed here

Acknowledgements
This dataset is an ensemble of data collected from TMDB and GroupLens.
The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This product uses the TMDb API but is not endorsed or certified by TMDb. Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. You can try it for yourself here.

The Movie Links and Ratings have been obtained from the Official GroupLens website. The files are a part of the dataset available here
'''