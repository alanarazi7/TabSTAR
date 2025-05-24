from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: product_sentiment_machine_hack
====
Examples: 5091
====
URL: https://www.openml.org/search?type=data&id=46651
====
Description: Classify the sentiment (4-way classification) of user reviews of products based on the review
    text and product type (e.g. Tablet, Mobile, etc.). Intuitively, we expect most of the predictive signal to
    lie in the text, but predictions can be further improved by accounting for the fact that certain types of
    products tend to receive certain user sentiment. Representing a relatively simple multimodal task
    with only a single text feature and one categorical feature, this dataset originally stems from a 2020
    MachineHack prediction competition: https://machinehack.com/hackathons/product_
    sentiment_classification_weekend_hackathon_19/overview
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
 
 
https://machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/overview

Analyzing sentiments related to various products such as Tablet, Mobile and various other gizmos can be fun and difficult especially when collected across various demographics around the world. In this weekend hackathon, we challenge the machinehackers community to develop a machine learning model to accurately classify various products into 4 different classes of sentiments based on the raw text review provided by the user. Analyzing these sentiments will not only help us serve the customers better but can also reveal lot of customer traits present/hidden in the reviews.

The sentiment analysis requires a lot to be taken into account mainly due to the preprocessing involved to represent raw text and make them machine-understandable. Usually, we stem and lemmatize the raw information and then represent it using TF-IDF, Word Embeddings, etc. However, provided the state-of-the-art NLP models such as Transformer based BERT models one can skip the manual feature engineering like TF-IDF and Count Vectorizers.

In this short span of time, we would encourage you to leverage the ImageNet moment (Transfer Learning) in NLP using various pre-trained models.

Attribute Description:

Text_ID - Unique Identifier
Product_Description - Description of the product review by a user
Product_Type - Different types of product (9 unique products)
Class - Represents various sentiments
0 - Cannot Say
1 - Negative
2 - Positive
3 - No Sentiment


====
Target Variable: Sentiment (numeric, 4 distinct): ['2', '3', '1', '0']
====
Features:

Unnamed: 0 (numeric, 5091 distinct): ['5743', '1305', '1009', '4475', '2903', '403', '1769', '1372', '3763', '4198']
Text_ID (numeric, 5091 distinct): ['2333', '6642', '7151', '4968', '2396', '4606', '2766', '5854', '74', '2293']
Product_Description (string, 5084 distinct): ["RT @mention \x89÷¼ Happy Woman's Day! Make love, not fuss! \x89÷_ {link} \x89ã_ #edchat #musedchat #sxsw #sxswi #classical #newTwitter", 'Win free ipad 2 from webdoc.com #sxsw RT', 'RT @mention \x89÷¼ GO BEYOND BORDERS! \x89÷_ {link} \x89ã_ #edchat #musedchat #sxsw #sxswi #classical #newTwitter', 'RT @mention Marissa Mayer: Google Will Connect the Digital &amp; Physical Worlds Through Mobile - {link} #sxsw', "RT @mention RT @mention It's not a rumor: Apple is opening up a temporary store in downtown Austin for #SXSW and the iPad 2 launch {link}", 'RT @mention Google to Launch Major New Social Network Called Circles, Possibly Today {link} #sxsw', 'RT @mention Marissa Mayer: Google Will Connect the Digital &amp; Physical Worlds Through Mobile - {link} #SXSW', 'On my way to the Apple Store to upgrade my pager. {link} #SXSW', "Mayer says it makes sense to condense Google's location products &amp; features now that experiments show which ones are successful #SxSW #SUxSW", "It's on, @mention just walked in to The Industry Party by #GSDM &amp; #Google Austin, TX. #SXSW"]
Product_Type (numeric, 10 distinct): ['9', '6', '2', '7', '3', '5', '8', '1', '0', '4']
'''

CONTEXT = "Product Sentiment Analysis"
TARGET = CuratedTarget(raw_name="Sentiment", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'2': 'Positive', '3': 'No Sentiment', '1': 'Negative', '0': 'Cannot Say'})
COLS_TO_DROP = ["Unnamed: 0", "Text_ID"]
FEATURES = []
