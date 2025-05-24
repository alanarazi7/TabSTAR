from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: stefanoleone992/filmtv-movies-dataset/filmtv_movies.csv
====
Examples: 41399
====
URL: https://www.kaggle.com/stefanoleone992/filmtv-movies-dataset/filmtv_movies.csv
====
Description: 
FilmTV movies dataset
41k+ movies scraped from FilmTV

About Dataset
Context
Movies data are available on websites such as IMDb with average votes, vote numbers, reviews and descriptions. While IMDb is the most trustworthy source for data, other websites as FilmTV.it can provide the information on how users from different countries rate the movies compared to each other.

Content
Each row represents a movie available on FilmTV.it, with the original title, year, genre, duration, country, director, actors, average vote and votes.
The fie in the English version contains 41,399 movies and 19 attributes, while the Italian version contains one extra-attribute for the local title used when the movie was published in Italy.

Acknowledgements
Data has been scraped from the publicly available website https://www.filmtv.it as of 2023-10-21.

Inspiration
To provide further information in regards to the aspects that make a movie successful from users or profit perspective, and it can be combined with other movie datasets publicly available (RottenTomatoes, etc.).

====
Features:

filmtv_id (int64, 41399 distinct): ['2', '52096', '52206', '52214', '52215', '52217', '52221', '52222', '52224', '52226']
title (object, 39531 distinct): ['Les Vampires', 'Pinocchio', 'Riget II', 'Dr. Jekyll and Mr. Hyde', 'Sibiriada', 'Little Women', 'The Phantom of the Opera', 'The Three Musketeers', 'Home', 'Anna Karenina']
year (int64, 115 distinct): ['2018', '2017', '2016', '2019', '2015', '2013', '2014', '2012', '2011', '2009']
genre (object, 30 distinct): ['Drama', 'Comedy', 'Thriller', 'Horror', 'Action', 'Documentary', 'Adventure', 'Western', 'Animation', 'Romantic']
duration (int64, 269 distinct): ['90', '100', '95', '105', '92', '93', '110', '85', '98', '96']
country (object, 2087 distinct): ['United States', 'Italy', 'France', 'Great Britain', 'Germany', 'Japan', 'Canada', 'Spain', 'Italy, France', 'South Korea']
directors (object, 14752 distinct): ['Steno', 'Mario Mattòli', 'Carlo Vanzina', 'John Ford', 'Umberto Lenzi', 'Lucio Fulci', 'Takashi Miike', 'Werner Herzog', 'Dino Risi', 'Ingmar Bergman']
actors (object, 39142 distinct): ['Attori non professionisti', 'Musidora, Edouard Mathé, Marcel Lévesque, Fernand Herrmann', 'Henry Arnold, Salome Kammer, Anke Sevenich, Noemi Steuer', 'Ernst-Hugo Järegård, Peter Mygind, Kirsten Rolffes, Søren Pilmark', "Vladimir Fogel, Natal'ja Glan, Igor Ilyinskij, Boris Barnet", 'Vladimir Sajmolov, Nikita Mikhalkov, Natalja Andrejcenko, Vitalji Solomin', 'Maury Chaykin, Timothy Hutton, Bill Smitrovich, Colin Fox', 'Vasco Rossi', 'Bud Spencer, Michael Winslow', 'Filippo Timi, Lucia Mascino, Enrica Guidi, Stefano Fresi, Corrado Guzzanti, Alessandro Benvenuti, Atos Davini, Marcello Marziali, Massimo Paganelli']
avg_vote (float64, 90 distinct): ['6.0', '6.3', '7.0', '6.5', '5.8', '5.0', '6.8', '5.5', '5.3', '7.3']
critics_vote (float64, 625 distinct): ['6.0', '4.0', '8.0', '7.0', '5.0', '3.0', '6.5', '5.5', '2.0', '4.5']
public_vote (float64, 10 distinct): ['6.0', '7.0', '5.0', '4.0', '8.0', '3.0', '9.0', '2.0', '10.0', '1.0']
total_votes (int64, 592 distinct): ['3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
description (object, 39751 distinct): ['Plot in preparation', 'The "Vampires" are the ferocious members of the criminal gang led by the beautiful and ruthless Irma Vep. The journalist Philippe Guérande, backed by a repentant "vampire", the gravedigger Mazamette ...', 'At the "Il Regno" hospital in Copenhagen, the chain of strange and disturbing events seems to never end: the apparitions of the ghost ambulance, the evil presences identified in the basements by Mrs. Drusse, and much more ...', '\xa0', '300', 'Plot in preparation.', 'Bob and Joe, two inmates in a Costa Rican maximum security prison, manage to escape and, in order not to be recognized by the police, are forced to pretend to be monks. They take refuge in the remote mission of San Rolando under the guise of Father Orso and Father Zaccaria.', '1', "Assisted by photographer Fogel 'and journalist Barnet, Tom Hopkins saves Vivian Mend from the clutches of Cice, who in the meantime is cultivating a monstrous criminal project ...", 'The anguish and problems, even family ones, of the pilot colonel Paul Tibbets who, at the end of the Second World War, was charged with the material execution of the secret project that led to the launch of the first atomic bomb on the Japanese city of Hiroshima. The bombing caused a real massacre and forced the Japanese to surrender, but in the foreground in the film is the conscience drama of Tibbets.']
notes (object, 18623 distinct): ['Inspired by a true story.', 'Based on a true story.', 'Inspired by real events.', 'Based on real events.', 'Adaptation of the novel of the same name by Andrea Camilleri.', 'A three-episode film that is a brilliant attempt to "find the cinematic equivalent of the Jack London committed novel".', 'Combination of two episodes of the TV series True Justice.', "Television adaptation of Agatha Christie's novel of the same name.", '\xa0', 'Two-part television drama.']
humor (int64, 6 distinct): ['0', '1', '2', '3', '4', '5']
rhythm (int64, 6 distinct): ['2', '0', '3', '1', '4', '5']
effort (int64, 6 distinct): ['0', '1', '2', '3', '4', '5']
tension (int64, 6 distinct): ['0', '1', '2', '3', '4', '5']
erotism (int64, 6 distinct): ['0', '1', '2', '3', '4', '5']
'''

CONTEXT = "FilmTV movies ataset rating"
TARGET = CuratedTarget(raw_name="avg_vote", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['filmtv_id']
FEATURES = []

DESCRIPTION = '''
FilmTV movies dataset
41k+ movies scraped from FilmTV

About Dataset
Context
Movies data are available on websites such as IMDb with average votes, vote numbers, reviews and descriptions. While IMDb is the most trustworthy source for data, other websites as FilmTV.it can provide the information on how users from different countries rate the movies compared to each other.

Content
Each row represents a movie available on FilmTV.it, with the original title, year, genre, duration, country, director, actors, average vote and votes.
The fie in the English version contains 41,399 movies and 19 attributes, while the Italian version contains one extra-attribute for the local title used when the movie was published in Italy.

Acknowledgements
Data has been scraped from the publicly available website https://www.filmtv.it as of 2023-10-21.

Inspiration
To provide further information in regards to the aspects that make a movie successful from users or profit perspective, and it can be combined with other movie datasets publicly available (RottenTomatoes, etc.).
'''