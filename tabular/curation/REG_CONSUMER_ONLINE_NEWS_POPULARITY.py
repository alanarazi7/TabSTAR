from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: news_popularity2
====
Examples: 24007
====
URL: https://www.openml.org/search?type=data&id=46662
====
Description: Predict the popularity (number of shares on social media, on log-scale) of Mashable.com news
    articles based on the text of their title, as well as auxiliary numerical features like the number of
    words in the article, its average token length, and how many keywords are listed, etc. This dataset
    represents a very difficult prediction problem with only weak signal offered by the observed features.
    It is fundamentally hard to forecast how popular an article will be based only on its title and crude
    numerical summary statistics. To be comprehensive, an AutoML benchmark should contain at least
    one challenging problem like this. While pop stems from the same original data source as channel,
    the two have different labels to predict and do not share exactly the same set of features.
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: log_shares (numeric, 1290 distinct): ['7.004', '7.0909', '7.1709', '7.2449', '7.3139', '6.9088', '7.3784', '7.439', '7.4961', '7.5501']
====
Features:

n_tokens_content (numeric, 2111 distinct): ['0.0', '286.0', '246.0', '281.0', '235.0', '215.0', '225.0', '317.0', '240.0', '242.0']
average_token_length (numeric, 19527 distinct): ['0.0', '4.5', '5.0', '4.6667', '4.75', '4.6', '4.4286', '4.7778', '4.625', '4.8']
num_keywords (numeric, 10 distinct): ['7', '6', '10', '8', '5', '9', '4', '3', '2', '1']
article_title (string, 23862 distinct): ['Viral Video Recap: Must-Watch Memes of the Week', 'Top 10 Tech This Week', 'Mashable', 'Must Reads: The #Longreads You Missed This Week  ', '5 Fascinating Facts We Learned From Reddit This Week', "6 Apps You Don't Want To Miss", 'Top 25 Digital Media Resources This Week', "7 Apps You Don't Want To Miss", 'Top 5 Apps for Kids This Week', 'Access Denied']
'''

CONTEXT = "Popularity of Online News"
TARGET = CuratedTarget(raw_name="log_shares", new_name="Number of Log Shares in social media",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []