from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: okcupid-stem
====
Examples: 50789
====
URL: https://www.openml.org/search?type=data&id=42734
====
Description: User profile data for San Francisco OkCupid users published in [Kim, A. Y., & Escobedo-Land, A. (2015). OKCupid data for introductory statistics and data science courses. Journal of Statistics Education, 23(2).]. The curated dataset was downloaded from [https://github.com/rudeboybert/JSE_OkCupid]. The original dataset was created with the use of a python script that pulled the data from public profiles on www.okcupid.com on 06/30/2012. It includes people (n = 59946) within a 25 mile radius of San Francisco, who were online in the last year (06/30/2011), with at least one profile picture. Permission to use this data was obtained by the author of the original paper from OkCupid president and co-founder Christian Rudder under the condition that the dataset remains public. As target, the variable 'job' was collapsed into three categories: 'stem', 'non_stem', and 'student'. STEM jobs were defined as 'job' %in% c('computer / hardware / software', 'science / tech / engineering'). Observations with 'job' %in% c('unemployed', 'retired', 'rather not say') or missing values in 'job' were removed. The original dataset also included ten open text variables 'essay0' to 'essay9', which were removed from the dataset uploaded here. The dataset further includes the date/time variable 'last_online' (ignored by default) which could be used to construct additional features. Using OkCupid data for predicting STEM jobs was inspired by Max Kuhns book 'Feature Engineering and Selection: A Practical Approach for Predictive Models' [https://bookdown.org/max/FES/].
====
Target Variable: job (nominal, 3 distinct): ['non_stem', 'stem', 'student']
====
Features:

age (numeric, 53 distinct): ['27', '26', '28', '25', '29', '30', '24', '31', '32', '23']
body_type (nominal, 13 distinct): ['average', 'fit', 'athletic', 'thin', 'curvy', 'a little extra', 'skinny', 'full figured', 'overweight', 'jacked']
diet (nominal, 19 distinct): ['mostly anything', 'anything', 'strictly anything', 'mostly vegetarian', 'mostly other', 'strictly vegetarian', 'vegetarian', 'strictly other', 'other', 'mostly vegan']
drinks (nominal, 7 distinct): ['socially', 'rarely', 'often', 'not at all', 'very often', 'desperately']
drugs (nominal, 4 distinct): ['never', 'sometimes', 'often']
education (nominal, 33 distinct): ['graduated from college/university', 'graduated from masters program', 'working on college/university', 'working on masters program', 'graduated from two-year college', 'graduated from high school', 'graduated from ph.d program', 'graduated from law school', 'working on two-year college', 'dropped out of college/university']
ethnicity (nominal, 209 distinct): ['white', 'asian', 'hispanic / latin', 'black', 'other', 'hispanic / latin, white', 'indian', 'asian, white', 'white, other', 'pacific islander']
height (numeric, 58 distinct): ['70.0', '68.0', '67.0', '72.0', '69.0', '71.0', '66.0', '64.0', '65.0', '73.0']
income (nominal, 13 distinct): ['20000', '100000', '80000', '30000', '40000', '50000', '60000', '70000', '150000', '1000000']
location (nominal, 184 distinct): ['san francisco, california', 'oakland, california', 'berkeley, california', 'san mateo, california', 'palo alto, california', 'alameda, california', 'emeryville, california', 'san rafael, california', 'hayward, california', 'redwood city, california']
offspring (nominal, 16 distinct): ['doesn&rsquo;t have kids', 'doesn&rsquo;t have kids, but might want them', 'doesn&rsquo;t have kids, but wants them', 'doesn&rsquo;t want kids', 'has kids', 'has a kid', 'doesn&rsquo;t have kids, and doesn&rsquo;t want any', 'has kids, but doesn&rsquo;t want more', 'has a kid, but doesn&rsquo;t want more', 'has a kid, and might want more']
orientation (nominal, 3 distinct): ['straight', 'gay', 'bisexual']
pets (nominal, 16 distinct): ['likes dogs and likes cats', 'likes dogs', 'likes dogs and has cats', 'has dogs', 'has dogs and likes cats', 'likes dogs and dislikes cats', 'has dogs and has cats', 'has cats', 'likes cats', 'has dogs and dislikes cats']
religion (nominal, 46 distinct): ['agnosticism but not too serious about it', 'other', 'agnosticism', 'agnosticism and laughing about it', 'catholicism but not too serious about it', 'other and laughing about it', 'atheism', 'atheism and laughing about it', 'christianity but not too serious about it', 'christianity']
sex (nominal, 2 distinct): ['m', 'f']
sign (nominal, 49 distinct): ['gemini and it&rsquo;s fun to think about', 'scorpio and it&rsquo;s fun to think about', 'leo and it&rsquo;s fun to think about', 'libra and it&rsquo;s fun to think about', 'taurus and it&rsquo;s fun to think about', 'cancer and it&rsquo;s fun to think about', 'aries and it&rsquo;s fun to think about', 'virgo and it&rsquo;s fun to think about', 'sagittarius and it&rsquo;s fun to think about', 'pisces and it&rsquo;s fun to think about']
smokes (nominal, 6 distinct): ['no', 'sometimes', 'when drinking', 'yes', 'trying to quit']
speaks (nominal, 7020 distinct): ['english', 'english (fluently)', 'english (fluently), spanish (poorly)', 'english (fluently), spanish (okay)', 'english (fluently), spanish (fluently)', 'english (fluently), french (poorly)', 'english, spanish', 'english, spanish (okay)', 'english, spanish (poorly)', 'english (fluently), chinese (fluently)']
status (nominal, 5 distinct): ['single', 'seeing someone', 'available', 'married', 'unknown']
'''

CONTEXT = "OKCupid Dating Profiles for Job Prediction"
TARGET  = CuratedTarget(raw_name="job", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []