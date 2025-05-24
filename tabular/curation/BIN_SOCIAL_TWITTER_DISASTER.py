from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Disaster-Tweets
====
Examples: 11370
====
URL: https://www.openml.org/search?type=data&id=43395
====
Description: Context
The file contains over 11,000 tweets associated with disaster keywords like crash, quarantine, and bush fires as well as the location and keyword itself. The data structure was inherited from Disasters on social media
The tweets were collected on Jan 14th, 2020.
Some of the topics people were tweeting:

The eruption of Taal Volcano in Batangas, Philippines
Coronavirus
Bushfires in Australia
Iran downing of the airplane flight PS752

Disclaimer: The dataset contains text that may be considered profane, vulgar, or offensive.
Inspiration
The intention was to enrich the already available data for this topic with newly collected and manually classified tweets.
The initial source Disasters on social media which is used in Real or Not? NLP with Disaster Tweets competition on Kaggle.
====
Features:

id (numeric, 11370 distinct): ['0', '7550', '7574', '7575', '7576', '7577', '7578', '7579', '7580', '7581']
keyword (string, 219 distinct): ['thunderstorm', 'flattened', 'mass20murder', 'stretcher', 'drown', 'sirens', 'drowning', 'engulfed', 'fear', 'obliterate']
location (string, 4358 distinct): ['United States', 'Australia', 'London, England', 'UK', 'India', 'London', 'United Kingdom', 'USA', 'California, USA', 'Los Angeles, CA']
text (string, 11220 distinct): ['I want to help you with my project to save the Caribbean Sea from floods and hurricanes https://t.co/qD8Om9NqQK', 'We wanted to entertain you all with a good movie on Sankranthi festival. And you gave us a Landslide Victory as return gift.', "Study? Don't you mean disinformation campaign? https://t.co/6UNSN6CzYq", 'ROAD TRAFFIC COLLISION Victoria Road Junction with Marathon Road. Road Partially Blocked. Police en route.', 'Gov. Wolf vows to veto bill loosening rules for conventional oil and gas wells https://t.co/T1qYcGo7TQ', '28 menstruation huts demolished in a single ward  https://t.co/7atgVQyYmj https://t.co/bR1VDytHFi', 'Widow of CIA agent killed in 2009 Afghanistan suicide bomb attack breaks her silence 10 years on https://t.co/GtCOCsBtLx', 'a high IQ play would be being bearish BTC for January 2020 should fall off a cliff soon https://t.co/v7SClFdWRc', 'Incident closed: Collision on M50 between J09 - RED and J07 - LUCAN (North) https://t.co/kt5EM5opsR', 'Crews are planning to build a bridge for cyclists and pedestrians near the trail. https://t.co/Fp4Yrp33L1']
target (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Disaster Tweets"
TARGET = CuratedTarget(raw_name="target", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["id'"]
FEATURES = []