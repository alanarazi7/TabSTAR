from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: Harry-Potter-fanfiction-data
====
Examples: 648493
====
URL: https://www.openml.org/search?type=data&id=43544
====
Description: Context
Huge Harry Potter fan. Wanted to collect fan-fiction data to make a dashboard and visualize it. Its in the works. 
Content
I scraped this data from https://www.fanfiction.net/book/Harry-Potter/ using requests and beautiful soup. The data is completely structured. The scraping code can be found at https://github.com/nt03/HarryPotter_fanfics/tree/master/ffnet
It contains all HP Fanfic entries written between 2001-2019 in all available languages. The data doesn't contain the story itself but just the story blurb.
Acknowledgements
The code is entirely mine. The thumbnail and banner are attributed to [Photo by Christian Wagner on Unsplash]
Inspiration
You can answer questions like 'which is the most popular pairing', which language has the most ffs written in it, what has been the general trend like since the last movie or book came out.
====
Features:

Chapters (numeric, 228 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Favs (string, 3483 distinct): [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', ' 10']
Follows (string, 3139 distinct): [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', ' 10']
Published (string, 5502 distinct): [' 3/31/2011', ' 7/25/2011', ' 7/24/2011', ' 7/22/2011', ' 7/19/2011', ' 7/23/2007', ' 7/18/2011', ' 7/24/2007', ' 8/3/2011', ' 8/8/2011']
Reviews (numeric, 2458 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
Words (string, 68086 distinct): [' 100', ' 535', ' 551', ' 542', ' 598', ' 561', ' 1,000', ' 500', ' 534', ' 545']
author (string, 156280 distinct): ['reviews', "DarylDixon'sgirl1985", 'Lomonaaeren', 'everlovingdeer', 'Ida59', 'phoenixgirl26', 'alyssialui', 'Clenery Aingremont', 'HP Slash Luv', 'Cheeky Slytherin Lass']
characters (string, 36517 distinct): ['Hermione G., Draco M.', 'Harry P., Draco M.', 'Draco M., Hermione G.', 'Harry P.', 'James P., Lily Evans P.', 'Draco M., Harry P.', 'Harry P., Hermione G.', 'Harry P., Ginny W.', 'Harry P., Severus S.', 'Lily Evans P., James P.']
genre (string, 403 distinct): ['Romance', 'Romance/Humor', 'Romance/Drama', 'Humor', 'Humor/Romance', 'Romance/Angst', 'Drama/Romance', 'Romance/Friendship', 'Romance/Hurt/Comfort', 'Drama']
language (string, 43 distinct): ['English', 'Spanish', 'French', 'Portuguese', 'German', 'Polish', 'Indonesian', 'Swedish', 'Dutch', 'Italian']
rating (string, 4 distinct): ['T', 'K+', 'M', 'K']
story_link (string, 648090 distinct): ['https://www.fanfiction.net/s/9092481/1/The-Forgotten-One', 'https://www.fanfiction.net/s/5626215/1/Maybe-That-ssss-Your-Problem-Too', 'https://www.fanfiction.net/s/13096467/1/My-Son-Born-in-The-Darkness', 'https://www.fanfiction.net/s/10572509/1/Te-enseC3B1arC3A9-que-es-vivir', 'https://www.fanfiction.net/s/11845295/1/What-is-the-Point', 'https://www.fanfiction.net/s/11865847/1/The-Regret', 'https://www.fanfiction.net/s/12542038/1/The-Black-Twins', 'https://www.fanfiction.net/s/2665794/1/Spinner', 'https://www.fanfiction.net/s/4036064/1/Hermione-s-Parents', 'https://www.fanfiction.net/s/12790626/1/Sea-of-lights-A-Snily-Fanfic']
synopsis (string, 647085 distinct): ['...', '-', '.', 'Requested.', 'The title says it all.', ' ', '-ON HIATUS-', 'DISCONTINUED.', 'Removed', 'Drabble']
title (string, 474188 distinct): ['Always', 'Memories', 'Broken', 'Secrets', 'Unexpected', 'Home', 'Perfect', 'Alone', 'Changes', 'Lost']
published_mmyy (string, 181 distinct): ['7-2011', '8-2011', '7-2007', '8-2007', '12-2011', '12-2010', '7-2009', '9-2011', '8-2009', '7-2005']
pairing (string, 9991 distinct): ['Draco M., Hermione G.', 'Harry P., Draco M.', 'Draco M., Harry P.', 'Hermione G., Draco M.', 'Harry P.', 'James P., Lily Evans P.', 'Harry P., Hermione G.', 'Harry P., Ginny W.', 'Lily Evans P., James P.', 'Hermione G., Ron W.']
'''

CONTEXT = "Harry Potter Fan Fiction Story"
TARGET = CuratedTarget(raw_name="rating", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['story_link', 'published_mmyy']
FEATURES = [CuratedFeature(raw_name="Published", feat_type=FeatureType.DATE)]