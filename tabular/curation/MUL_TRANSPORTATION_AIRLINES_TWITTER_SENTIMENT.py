from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Airlines-Tweets-Sentiments
====
Examples: 1097
====
URL: https://www.openml.org/search?type=data&id=43397
====
Description: Context
A dataset I used to classify tweets about my company.
I took tweets and I classified them manually as positive, negative or neutral.
Content
There are 4 columns :
Id : the tweed id, unique.
tweettext : the tweet
tweetlang : always EN, all tweets are in english
tweetsentiment_value : 0 for negative, 1 for neutral, 2 for positive
Acknowledgements
Do what you want with it.
Inspiration
The aim of this dataset is to be able to determine if a tweet is positive or negative about an airline company.
====
Features:

_id (string, 1097 distinct): ['595e60b48fcd022a715f7b7b', '5963b50f4fe31f4f52a022cb', '5963b73b4fe31f4f52a022d6', '5963b6af4fe31f4f52a022d5', '5963b6464fe31f4f52a022d2', '5963b6354fe31f4f52a022d1', '5963b5ce4fe31f4f52a022ce', '5963b56d4fe31f4f52a022cc', '5963b5064fe31f4f52a022ca', '5963e90f4fe31f4f52a02359']
tweet_text (string, 1090 distinct): ['airfrance thank you', 'air france sells servair stake to hna group https://t.co/9jozjeckzn traveltip airline travelnews travel news', "airfrance you made our kid fly in horrible condition and you and your partner aireuropa don't even reply to our complaint!\nwhat a shame", 'airfrance have a good weekend', 'airfrance thanks!', 'air france a380 - an impossible engineering documentary https://t.co/z26l7uv1bp via youtube', 'alaskaair yes. called air france too and they said they are no longer a partner. was the only reason i booked on t https://t.co/jtjy4bu5zm', 'air france a330-200 f-gzck lining up on houston iah 33l for departure \navgeek airbus airbus330 airfrance https://t.co/wttme3buqc', 'quasi come 747 airfrance in atterraggio a saintmartin... https://t.co/l7ha3tzefh', 'this is random.\n\nairfranceus is the official airline sponsor of the thesfmarathon: https://t.co/znecgdgc4r']
tweet_lang (string, 1 distinct): ['en']
tweet_sentiment_value (numeric, 3 distinct): ['1', '0', '2']
'''

CONTEXT = "Airlines Tweets Sentiments"
TARGET = CuratedTarget(raw_name="tweet_sentiment_value", new_name="Tweet Sentiment",
                       task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"0": "Negative", "1": "Neutral", "2": "Positive"})
COLS_TO_DROP = ['_id', 'tweet_lang']
FEATURES = []