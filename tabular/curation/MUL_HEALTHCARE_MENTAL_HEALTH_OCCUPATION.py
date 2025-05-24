from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: Mental_Health_Dataset
====
Examples: 292364
====
URL: https://www.openml.org/search?type=data&id=46721
====
Description: About Dataset
This dataset appears to contain a variety of features related to text analysis, sentiment analysis, and psychological indicators, likely derived from posts or text data. Some features include readability indices such as Automated Readability Index (ARI), Coleman Liau Index, and Flesch-Kincaid Grade Level, as well as sentiment analysis scores like sentiment compound, negative, neutral, and positive scores. Additionally, there are features related to psychological aspects such as economic stress, isolation, substance use, and domestic stress. The dataset seems to cover a wide range of linguistic, psychological, and behavioural attributes, potentially suitable for analyzing mental health-related topics in online communities or text data.

Benefits of using this dataset:
Insight into Mental Health: The dataset provides valuable insights into mental health by analyzing linguistic patterns, sentiment, and psychological indicators in text data. Researchers and data scientists can gain a better understanding of how mental health issues manifest in online communication.
Predictive Modeling: With a wide range of features, including sentiment analysis scores and psychological indicators, the dataset offers opportunities for developing predictive models to identify or predict mental health outcomes based on textual data. This can be useful for early intervention and support.
Community Engagement: Mental health is a topic of increasing importance, and this dataset can foster community engagement on platforms like Kaggle. Data enthusiasts, researchers, and mental health professionals can collaborate to analyze the data and develop solutions to address mental health challenges.
Data-driven Insights: By analyzing the dataset, users can uncover correlations and patterns between linguistic features, sentiment, and mental health indicators. These insights can inform interventions, policies, and support systems aimed at promoting mental well-being.
Educational Resource: The dataset can serve as a valuable educational resource for teaching and learning about mental health analytics, sentiment analysis, and text mining techniques. It provides a real-world dataset for students and practitioners to apply data science skills in a meaningful context.

https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset/data

Columns
'Timestamp', 'Gender', 'Country', 'Occupation' (target), 'self_employed',
       'family_history', 'treatment', 'Days_Indoors', 'Growing_Stress',
       'Changes_Habits', 'Mental_Health_History', 'Mood_Swings',
       'Coping_Struggles', 'Work_Interest', 'Social_Weakness',
       'mental_health_interview', 'care_options'
====
Target Variable: Occupation (string, 5 distinct): ['Housewife', 'Student', 'Corporate', 'Others', 'Business']
====
Features:

Timestamp (string, 580 distinct): ['8/27/2014 11:43', '8/27/2014 12:31', '8/27/2014 12:53', '8/27/2014 16:21', '8/27/2014 12:39', '8/27/2014 12:48', '8/27/2014 12:34', '8/27/2014 12:33', '8/27/2014 15:23', '8/27/2014 14:13']
Gender (string, 2 distinct): ['Male', 'Female']
Country (string, 35 distinct): ['United States', 'United Kingdom', 'Canada', 'Australia', 'Netherlands', 'Ireland', 'Germany', 'Sweden', 'India', 'France']
self_employed (string, 2 distinct): ['No', 'Yes']
family_history (string, 2 distinct): ['No', 'Yes']
treatment (string, 2 distinct): ['Yes', 'No']
Days_Indoors (string, 5 distinct): ['1-14 days', '31-60 days', 'Go out Every day', 'More than 2 months', '15-30 days']
Growing_Stress (string, 3 distinct): ['Maybe', 'Yes', 'No']
Changes_Habits (string, 3 distinct): ['Yes', 'Maybe', 'No']
Mental_Health_History (string, 3 distinct): ['No', 'Maybe', 'Yes']
Mood_Swings (string, 3 distinct): ['Medium', 'Low', 'High']
Coping_Struggles (string, 2 distinct): ['No', 'Yes']
Work_Interest (string, 3 distinct): ['No', 'Maybe', 'Yes']
Social_Weakness (string, 3 distinct): ['Maybe', 'No', 'Yes']
mental_health_interview (string, 3 distinct): ['No', 'Maybe', 'Yes']
care_options (string, 3 distinct): ['No', 'Yes', 'Not sure']
'''

CONTEXT = "Mental Health derived from text data and psychological indicators"
TARGET = CuratedTarget(raw_name='Occupation', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Timestamp", feat_type=FeatureType.DATE)]