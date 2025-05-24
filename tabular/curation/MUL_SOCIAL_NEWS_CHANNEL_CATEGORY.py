from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: news_channel
====
Examples: 20284
====
URL: https://www.openml.org/search?type=data&id=46652
====
Description: Predict which news category (i.e. channel) a Mashable.com news article belongs to based
    on the text of its title, as well as auxiliary numerical features like the number of words in the article,
    its average token length, how many keywords are listed, etc. Representing a task with one text
    field but many tabular (numeric) features, the original version of this dataset was collected by [20]:
    https://archive.ics.uci.edu/ml/datasets/online+news+popularity
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: channel (string, 6 distinct): [' data_channel_is_world', ' data_channel_is_tech', ' data_channel_is_entertainment', ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_lifestyle']
====
Features:

article_title (string, 20169 distinct): ['Top 10 Tech This Week', 'Mashable', 'Must Reads: The #Longreads You Missed This Week  ', '5 Fascinating Facts We Learned From Reddit This Week', 'Viral Video Recap: Must-Watch Memes of the Week', "6 Apps You Don't Want To Miss", "7 Apps You Don't Want To Miss", 'Top 25 Digital Media Resources This Week', 'Top 5 Apps for Kids This Week', '25 Digital Media Resources You May Have Missed']
'''

CONTEXT = "News Channel Prediction"
TARGET = CuratedTarget(raw_name="channel", new_name="News Category", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={' data_channel_is_world': 'World',
                                      ' data_channel_is_tech': 'Tech',
                                      ' data_channel_is_entertainment': 'Entertainment',
                                      ' data_channel_is_bus': 'Business',
                                      ' data_channel_is_socmed': 'Social Media',
                                      ' data_channel_is_lifestyle': 'Lifestyle'})
COLS_TO_DROP = []
FEATURES = []
