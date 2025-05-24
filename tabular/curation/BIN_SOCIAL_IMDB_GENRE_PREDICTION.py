from tabular.datasets.manual_curation_obj import CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: imdb_genre_prediction
====
Examples: 800
====
URL: https://www.openml.org/search?type=data&id=46667
====
Description: Predict whether or not a movie falls within the Drama category based on text features like
    its name, description, actors/directors, and numerical features like its release year, runtime, etc.
    Representing a task with smaller sample-size, the original version of this dataset was collected from
    IMDB (the most popular movies): https://www.kaggle.com/PromptCloudHQ/imdb-data
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: Genre_is_Drama (nominal, 2 distinct): ['1', '0']
====
Features:

Rank (numeric, 800 distinct): ['161', '103', '953', '650', '473', '437', '172', '770', '380', '760']
Title (string, 799 distinct): ['The Host', 'Mine', 'The Martian', 'Sex and the City 2', 'Vampire Academy', '2012', 'The Longest Ride', 'Kubo and the Two Strings', 'High-Rise', 'The Expendables 3']
Description (string, 800 distinct): ['After a failed assassination attempt, a soldier finds himself stranded in the desert. Exposed to the elements, he must survive the dangers of the desert and battle the psychological and physical tolls of the treacherous conditions.', 'An astronaut becomes stranded on Mars after his team assume him dead, and must rely on his ingenuity to find a way to signal to Earth that he is alive.', "While wrestling with the pressures of life, love, and work in Manhattan, Carrie, Miranda, and Charlotte join Samantha for a trip to Abu Dhabi (United Arab Emirates), where Samantha's ex is filming a new movie.", 'Rose Hathaway is a Dhampir, half human-half vampire, a guardian of the Moroi, peaceful, mortal vampires living discreetly within our world. Her calling is to protect the Moroi from bloodthirsty, immortal Vampires, the Strigoi.', 'A frustrated writer struggles to keep his family alive when a series of global catastrophes threatens to annihilate mankind.', 'The lives of a young couple intertwine with a much older man, as he reflects back on a past love.', 'A young boy named Kubo must locate a magical suit of armour worn by his late father in order to defeat a vengeful spirit from the past.', 'Life for the residents of a tower block begins to run out of control.', 'Barney augments his team with new blood for a personal battle: to take down Conrad Stonebanks, the Expendables co-founder and notorious arms trader who is hell bent on wiping out Barney and every single one of his associates.', 'A look at the early years of boxer "Irish" Micky Ward and his brother who helped train him before going pro in the mid 1980s.']
Director (string, 559 distinct): ['Ridley Scott', 'David Yates', 'Michael Bay', 'Antoine Fuqua', 'M. Night Shyamalan', 'J.J. Abrams', 'Paul W.S. Anderson', 'Gore Verbinski', 'Peter Jackson', 'Guillermo del Toro']
Actors (string, 797 distinct): ['Gerard Butler, Aaron Eckhart, Morgan Freeman,Angela Bassett', 'Jennifer Lawrence, Josh Hutcherson, Liam Hemsworth, Woody Harrelson', 'Daniel Radcliffe, Emma Watson, Rupert Grint, Michael Gambon', 'Armie Hammer, Annabelle Wallis,Tom Cullen, Clint Dyer', 'Denzel Washington, Marton Csokas, Chloë Grace Moretz, David Harbour', 'John Cusack, Thandie Newton, Chiwetel Ejiofor,Amanda Peet', 'Scott Eastwood, Britt Robertson, Alan Alda, Jack Huston', 'Charlize Theron, Art Parkinson, Matthew McConaughey, Ralph Fiennes', 'Tom Hiddleston, Jeremy Irons, Sienna Miller, Luke Evans', 'Sylvester Stallone, Jason Statham, Jet Li, Antonio Banderas']
Year (numeric, 11 distinct): ['2016', '2015', '2014', '2013', '2010', '2011', '2012', '2007', '2009', '2008']
Runtime (Minutes) (numeric, 91 distinct): ['108', '118', '106', '117', '123', '100', '110', '112', '97', '104']
Rating (numeric, 52 distinct): ['6.7', '7.1', '6.6', '7.0', '6.5', '6.3', '7.3', '7.2', '7.8', '6.2']
Votes (numeric, 798 distinct): ['1427', '291', '5926', '556097', '44111', '297984', '58421', '72778', '25928', '137568']
Revenue (Millions) (numeric, 657 distinct): ['0.03', '0.01', '0.02', '0.32', '0.15', '0.05', '1.36', '0.92', '0.54', '6.86']
Metascore (numeric, 83 distinct): ['72.0', '64.0', '68.0', '48.0', '58.0', '51.0', '66.0', '76.0', '56.0', '57.0']
'''

CONTEXT = "IMDB Movies Genre Prediction"
TARGET = CuratedTarget(raw_name="Genre_is_Drama", new_name="Genre", task_type=SupervisedTask.BINARY,
                       label_mapping={'True': 'Drama', 'False': 'Not Drama'})
COLS_TO_DROP = []
FEATURES = []
