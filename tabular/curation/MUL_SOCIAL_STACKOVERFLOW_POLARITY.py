from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: StackOverflow-polarity
====
Examples: 4423
====
URL: https://www.openml.org/search?type=data&id=43160
====
Description: A Gold Standard of ~4,400 questions, answers, and comments from Stack Overflow, manually annotated for polarity. The dataset has been used for developing the EMTk toolkit for polarity detection from technical text.
====
Target Variable: polarity (nominal, 3 distinct): ['neutral', 'positive', 'negative']
====
Features:

text (string, 4333 distinct): ['Excellent, thanks!', 'Great question!', 'if the 2 structures variable are initialied with calloc or they are set with 0 by memset so you can compare your 2 structures with memcmp and there is no worry about structure garbage and this will allow you to earn time', 'You can use', 'Excellent answer!', '<3 <3 <3 ! ! ! :)', 'Too slow and I prefer to operate in the same mode as my users. Also, I just really hate switching to debug. And its a waste of diskspace.', 'You need to make the following changes to your code: However keep in mind that this is extremely vulnerable to SQL injection, and you should also swap MySQL for PDO/MySQLi immediately if you plan to use this code on a live website.', "I found an excellent tutorial on how to create site columns and content types - here : (THANKS AGAIN ROB!) Does anyone know of a written or video tutorial that will explain how to create a list and list instance in MOSS 2007. I use WSP Builder, and the build in templates from Microsoft aren't compatible (or up to the task). Preferably a method that focuses on the CAML (xml) , and explains in detail the theory, and demonstrates how it is done. Thank you.", "Gosh ! you're right !!! Excellent !!!! Is there a way to expand it programatically ? I added .setOngoing(true) (to the top) but it's not expanded... (not really the top because the first one is a notification when I'm plugged to USB for tools developpement ). Thank you again my friend !"]
'''

CONTEXT = "StackOverflow Polarity Detection"
TARGET = CuratedTarget(raw_name="polarity", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []