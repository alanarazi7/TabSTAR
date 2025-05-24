from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Binary-Dataset-of-Phishing-and-Legitimate-URLs
====
Examples: 11000
====
URL: https://www.openml.org/search?type=data&id=43622
====
Description: Description
The data set is provided csv file which provides the following resources that can be used as inputs for model building :
A collection of website URLs for 11001 websites. Each sample has 15 website parameters and a class label identifying it as a phishing website or not (0 or 1).
If URLs is Phished then label is 0 and for legitimate label is 1
The data set also serves as an input for project scoping and tries to specify the functional and non-functional requirements for it.
====
Features:

whois_regDate (numeric, 4038 distinct): ['-1', '1638', '92', '91', '5335', '90', '89', '93', '97', '7234']
whois_expDate (numeric, 1872 distinct): ['-1', '552', '273', '272', '143', '274', '275', '271', '267', '276']
whois_updatedDate (numeric, 1146 distinct): ['-1', '1310', '398', '86', '90', '89', '91', '84', '197', '50']
dot_count (numeric, 19 distinct): ['1', '2', '3', '4', '5', '6', '8', '7', '24', '9']
url_len (numeric, 347 distinct): ['11', '10', '12', '9', '13', '14', '8', '15', '7', '16']
digit_count (numeric, 128 distinct): ['0', '3', '1', '2', '4', '5', '9', '29', '6', '17']
special_count (numeric, 35 distinct): ['0', '2', '1', '5', '3', '7', '9', '4', '8', '6']
hyphen_count (numeric, 32 distinct): ['0', '1', '2', '3', '4', '5', '22', '6', '19', '21']
double_slash (numeric, 6 distinct): ['0', '1', '2', '5', '6', '3']
single_slash (numeric, 25 distinct): ['0', '3', '4', '2', '7', '5', '6', '8', '9', '10']
at_the_rate (numeric, 4 distinct): ['0', '1', '2', '3']
protocol (numeric, 2 distinct): ['0', '1']
protocol_count (numeric, 6 distinct): ['0', '1', '2', '5', '6', '3']
web_traffic (numeric, 2 distinct): ['0', '1']
label (numeric, 2 distinct): ['1', '0']
'''

CONTEXT = "URL Websites for Phishing Detection"
TARGET = CuratedTarget(raw_name="label", new_name="Is Website Legit", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
