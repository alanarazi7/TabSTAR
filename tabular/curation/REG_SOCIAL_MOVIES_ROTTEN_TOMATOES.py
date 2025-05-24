from typing import Any

from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: REG_SOCIAL_MOVIES_ROTTEN_TOMATOES
====
Examples: 7390
====
URL: http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv
====
Description: 
Rotten Tomatoes
(http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv)
Contain information on movies that can be found in Rotten Tomatoes movie rating website. The task is to predict the rating values of the movies.

====
Features:

Id (object, 7390 distinct): ['tt0054215', 'tt1478338', 'tt0342167', 'tt3713166', 'tt1570728', 'tt1702439', 'tt0431308', 'tt1282140', 'tt0450259', 'tt0163978']
Name (object, 7198 distinct): ['Treasure Island', 'Little Women', 'Jack and the Beanstalk', 'Dr. Jekyll and Mr. Hyde', 'The Last of the Mohicans', 'The Prisoner of Zenda', 'Fair Game', 'Beauty and the Beast', 'A Dog of Flanders', 'The Jungle Book']
Year (int64, 107 distinct): ['2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2006', '2007']
Release Date (object, 4795 distinct): ['2016 (USA)', '2015 (USA)', '14 August 2015 (USA)', '2010 (USA)', '25 September 2015 (USA)', '18 September 2015 (USA)', '21 August 2015 (USA)', '17 October 2014 (USA)', '16 October 2015 (USA)', '15 May 2015 (USA)']
Director (object, 3723 distinct): ['Woody Allen', 'Clint Eastwood', 'Steven Spielberg', 'Martin Scorsese', 'Billy Wilder', 'Oliver Stone', 'Alfred Hitchcock', 'Michael Curtiz', 'Ridley Scott', 'Francis Ford Coppola']
Creator (object, 6170 distinct): ['Woody Allen', 'John Hughes', 'John Waters', 'Kevin Smith', 'Joel Coen,Ethan Coen', 'M. Night Shyamalan', 'Jason Friedberg,Aaron Seltzer', 'Hal Hartley', 'Michael Moore', 'Frances Goodrich,Albert Hackett']
Actors (object, 7213 distinct): ['Daniel Radcliffe,Emma Watson,Rupert Grint', 'Winston Hibler', 'Groucho Marx,Chico Marx,Harpo Marx', 'William Shatner,Leonard Nimoy,DeForest Kelley', 'Bing Crosby,Bob Hope,Dorothy Lamour', 'Groucho Marx,Harpo Marx,Chico Marx', 'Sylvester Stallone,Talia Shire,Burt Young', 'Jennifer Lawrence,Josh Hutcherson,Liam Hemsworth', 'Kristen Stewart,Robert Pattinson,Taylor Lautner', 'Mark Hamill,Harrison Ford,Carrie Fisher']
Cast (object, 7307 distinct): ['Winston Hibler', "Ian McKellen,Martin Freeman,Richard Armitage,Ken Stott,Graham McTavish,William Kircher,James Nesbitt,Stephen Hunter,Dean O'Gorman,Aidan Turner,John Callen,Peter Hambleton,Jed Brophy,Mark Hadlow,Adam Brown", 'Kurt Russell,Zoë Bell,Rosario Dawson,Vanessa Ferlito,Sydney Tamiia Poitier,Tracie Thoms,Rose McGowan,Jordan Ladd,Mary Elizabeth Winstead,Quentin Tarantino,Marcy Harriell,Eli Roth,Omar Doom,Michael Bacall,Monica Staggs', 'Anthony Perkins,Vera Miles,John Gavin,Janet Leigh,Martin Balsam,John McIntire,Simon Oakland,Frank Albertson,Patricia Hitchcock,Vaughn Taylor,Lurene Tuttle,John Anderson,Mort Mills', 'M.C. Gainey,Paul Soter,Erik Stolhanske,Cloris Leachman,Jürgen Prochnow,Cameron Scher,Owain Yeoman,Tom Tate,Allan Graf,Chris Moss,Bjorn Johnson,Kevin Heffernan,Jay Chandrasekhar,Steve Lemme,Collin Thornton', 'Emma Stone,Penn Badgley,Amanda Bynes,Dan Byrd,Thomas Haden Church,Patricia Clarkson,Cam Gigandet,Lisa Kudrow,Malcolm McDowell,Aly Michalka,Stanley Tucci,Fred Armisen,Juliette Goglia,Jake Sandvig,Morgan Rusler', 'Leonardo DiCaprio,Djimon Hounsou,Jennifer Connelly,Kagiso Kuypers,Arnold Vosloo,Antony Coleman,Benu Mabhena,Anointing Lukola,David Harewood,Basil Wallace,Jimi Mistry,Michael Sheen,Marius Weyers,Stephen Collins,Ntare Guma Mbaho Mwine', "Leonardo DiCaprio,Daniel York,Patcharawan Patarakijjanon,Virginie Ledoyen,Guillaume Canet,Robert Carlyle,Somboon Phutaroth,Weeratham 'Norman' Wichairaksakui,Jak Boon,Peter Youngblood Hills,Jerry Swindall,Krongthong Thampradith,Apichart Chusakul,Sanya 'Gai' Cheunjit,Kaneung 'Nueng' Kenia", 'Lisa Adam,Frank Aldridge,Amitabh Bachchan,Steve Bisley,Richard Carter,Jason Clarke,Adelaide Clemens,Vince Colosimo,Max Cullen,Mal Day,Elizabeth Debicki,Leonardo DiCaprio,Joel Edgerton,Emmanuel Ekwenski,Eden Falk', 'Paul Sanchez,Lari White,Leonid Citer,David Allen Brooks,Yelena Popovic,Valentina Ananina,Semion Sudarikov,Tom Hanks,Peter Von Berg,Dmitri S. Boudrine,François Duhamel,Michael Forest,Viveka Davis,Nick Searcy,Jennifer Choe']
Language (object, 502 distinct): ['English', 'English,Spanish', 'English,French', 'English,German', 'English,Italian', 'English,Russian', 'English,Japanese', 'English,Mandarin', 'English,Arabic', 'English,Latin']
Country (object, 400 distinct): ['USA', 'UK,USA', 'USA,UK', 'USA,Germany', 'USA,Canada', 'Canada,USA', 'USA,Australia', 'USA,France', 'Germany,USA', 'France,USA']
Duration (object, 162 distinct): ['90 min', '93 min', '95 min', '91 min', '100 min', '92 min', '96 min', '97 min', '94 min', '88 min']
RatingValue (float64, 82 distinct): ['6.7', '6.4', '7.2', '6.5', '6.8', '7.1', '6.6', '7.0', '6.2', '7.3']
RatingCount (object, 5902 distinct): ['21', '30', '15', '20', '18', '26', '16', '9', '13', '239']
ReviewCount (object, 5292 distinct): ['1 user', '2 user', '1 critic', '3 user', '1 user,1 critic', '4 user,1 critic', '3 user,1 critic', '4 user,3 critic', '4 user', '2 critic']
Genre (object, 636 distinct): ['Drama', 'Comedy', 'Comedy,Romance', 'Comedy,Drama,Romance', 'Horror', 'Documentary', 'Comedy,Drama', 'Drama,Romance', 'Horror,Thriller', 'Action,Crime,Drama']
Filming Locations (object, 3190 distinct): ['Los Angeles, California, USA', 'New York City, New York, USA', 'Santa Clarita, California, USA', 'Metro-Goldwyn-Mayer Studios - 10202 W. Washington Blvd., Culver City, California, USA', 'California, USA', 'Warner Brothers Burbank Studios - 4000 Warner Boulevard, Burbank, California, USA', 'Universal Studios - 100 Universal City Plaza, Universal City, California, USA', 'Chicago, Illinois, USA', 'Paramount Studios - 5555 Melrose Avenue, Hollywood, Los Angeles, California, USA', 'Atlanta, Georgia, USA']
Description (object, 7261 distinct): ['Add a Plot', 'The plot is unknown.', 'Plot is unknown.', 'The plot is unknown at this time.', 'Peaceable Kingdom: The Journey Home explores the powerful struggle of conscience experienced by several people from traditional farming backgrounds who come to question the basic ...', 'Huckleberry Finn, a rambunctious boy adventurer chafing under the bonds of civilization, escapes his humdrum world and his selfish, plotting father by sailing a raft down the Mississippi ...', "John Groberg, a farm kid from Idaho Falls, crosses an ocean to become a missionary in the remote and exotic Tongan islands during the 1950's.", 'In rural 1800s England things go bad for a young matchmaker after she finds a man for another woman.', 'A thirty-something former child star hires a foster family to re-create the childhood he never had.', "While settling his recently deceased father's estate, a salesman discovers he has a sister whom he never knew about, leading both siblings to re-examine their perceptions about family and life choices."]
'''

def cure_duration(duration: Any) -> int:
    if isinstance(duration, str) and duration.endswith("min"):
        return int(duration.replace("min", "").strip())
    return duration

CONTEXT = "Rotten Tomatoes Movie Ratings"
TARGET = CuratedTarget(raw_name="RatingValue", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Id"]
FEATURES = [CuratedFeature(raw_name="Duration", feat_type=FeatureType.NUMERIC, processing_func=cure_duration)]

DESCRIPTION = '''
Rotten Tomatoes
(http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv)
Contain information on movies that can be found in Rotten Tomatoes movie rating website. The task is to predict the rating values of the movies.
'''