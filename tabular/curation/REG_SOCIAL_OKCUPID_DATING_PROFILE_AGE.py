from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: dating_profile
====
URL: https://www.openml.org/search?type=data&id=42164
====
Description: Anonymized data of dating profiles from OkCupid
====
Target Variable: age (numeric, 54 distinct): ['26', '27', '28', '25', '29', '24', '30', '31', '23', '32']
====
Features:

body_type (string, 13 distinct): ['average', 'fit', 'athletic', 'thin', 'curvy', 'a little extra', 'skinny', 'full figured', 'overweight', 'jacked']
diet (string, 19 distinct): ['mostly anything', 'anything', 'strictly anything', 'mostly vegetarian', 'mostly other', 'strictly vegetarian', 'vegetarian', 'strictly other', 'mostly vegan', 'other']
drinks (string, 7 distinct): ['socially', 'rarely', 'often', 'not at all', 'very often', 'desperately']
drugs (string, 4 distinct): ['never', 'sometimes', 'often']
education (string, 33 distinct): ['graduated from college/university', 'graduated from masters program', 'working on college/university', 'working on masters program', 'graduated from two-year college', 'graduated from high school', 'graduated from ph.d program', 'graduated from law school', 'working on two-year college', 'dropped out of college/university']
essay0 (string, 54350 distinct): ['.', 'ask me', 'under construction', '...', 'hi', 'hi!', 'just ask', 'hello!', "i'm awesome.", 'coming soon...']
essay1 (string, 51517 distinct): ['enjoying it.', 'living it.', 'living it', 'living it!', 'enjoying it!', 'enjoying it', 'living', 'living the dream.', 'working', 'living it to the fullest.']
essay2 (string, 48635 distinct): ['listening', 'making people laugh', 'making people laugh.', 'listening.', 'cooking', 'everything', 'making people smile.', 'everything.', 'sports', 'procrastinating']
essay3 (string, 43533 distinct): ['my smile', 'my eyes', 'my smile.', 'my eyes.', 'smile', 'eyes', 'you tell me.', 'my smile :)', 'my height', 'my hair']
essay4 (string, 49261 distinct): ['ask me', 'too many to list.', 'too many to list', 'ask me.', 'everything', 'yes.', 'too many to mention.', '.', 'dexter', 'too many']
essay5 (string, 48963 distinct): ['ask me', 'my family', 'family', 'my senses.', 'you', 'my fingers.', 'love', 'nothing', 'only six?', 'i hate this question.']
essay6 (string, 43603 distinct): ['my future', 'the future', 'life', 'my future.', 'everything', 'the future.', 'everything.', 'life.', 'traveling', 'my next adventure']
essay7 (string, 45554 distinct): ['out with friends', 'working', 'hanging out with friends', 'out with friends.', 'hanging out with friends.', 'working.', 'out and about', 'out', 'out.', 'out and about.']
essay8 (string, 39324 distinct): ['ask me', 'nothing', '...', 'ask me.', 'nothing.', "i'm on okcupid.", 'nope.', 'ask me in person.', 'really?', "i'll tell you later."]
essay9 (string, 45443 distinct): ['you want to.', 'you want to', 'you feel like it.', 'you want to!', 'you feel like it', "you're interested.", 'you want to know more.', 'you want.', "you're awesome.", "you're interested"]
ethnicity (string, 218 distinct): ['white', 'asian', 'hispanic / latin', 'black', 'other', 'hispanic / latin, white', 'indian', 'asian, white', 'white, other', 'pacific islander']
height (numeric, 63 distinct): ['70.0', '68.0', '67.0', '72.0', '69.0', '71.0', '66.0', '64.0', '65.0', '73.0']
income (numeric, 13 distinct): ['-1', '20000', '100000', '80000', '30000', '40000', '50000', '60000', '70000', '150000']
job (string, 22 distinct): ['other', 'student', 'science / tech / engineering', 'computer / hardware / software', 'artistic / musical / writer', 'sales / marketing / biz dev', 'medicine / health', 'education / academia', 'executive / management', 'banking / financial / real estate']
last_online (string, 30123 distinct): ['2012-06-29-22-56', '2012-06-30-21-51', '2012-06-30-22-09', '2012-06-30-22-56', '2012-06-30-23-27', '2012-06-30-23-55', '2012-06-30-22-53', '2012-06-30-22-57', '2012-06-30-10-15', '2012-06-30-22-50']
location (string, 199 distinct): ['san francisco, california', 'oakland, california', 'berkeley, california', 'san mateo, california', 'palo alto, california', 'alameda, california', 'san rafael, california', 'hayward, california', 'emeryville, california', 'redwood city, california']
offspring (string, 16 distinct): ['doesn&rsquo;t have kids', 'doesn&rsquo;t have kids, but might want them', 'doesn&rsquo;t have kids, but wants them', 'doesn&rsquo;t want kids', 'has kids', 'has a kid', 'doesn&rsquo;t have kids, and doesn&rsquo;t want any', 'has kids, but doesn&rsquo;t want more', 'has a kid, but doesn&rsquo;t want more', 'has a kid, and might want more']
orientation (string, 3 distinct): ['straight', 'gay', 'bisexual']
pets (string, 16 distinct): ['likes dogs and likes cats', 'likes dogs', 'likes dogs and has cats', 'has dogs', 'has dogs and likes cats', 'likes dogs and dislikes cats', 'has dogs and has cats', 'has cats', 'likes cats', 'has dogs and dislikes cats']
religion (string, 46 distinct): ['agnosticism', 'other', 'agnosticism but not too serious about it', 'agnosticism and laughing about it', 'catholicism but not too serious about it', 'atheism', 'other and laughing about it', 'atheism and laughing about it', 'christianity', 'christianity but not too serious about it']
sex (string, 2 distinct): ['m', 'f']
sign (string, 49 distinct): ['gemini and it&rsquo;s fun to think about', 'scorpio and it&rsquo;s fun to think about', 'leo and it&rsquo;s fun to think about', 'libra and it&rsquo;s fun to think about', 'taurus and it&rsquo;s fun to think about', 'cancer and it&rsquo;s fun to think about', 'pisces and it&rsquo;s fun to think about', 'sagittarius and it&rsquo;s fun to think about', 'virgo and it&rsquo;s fun to think about', 'aries and it&rsquo;s fun to think about']
smokes (string, 6 distinct): ['no', 'sometimes', 'when drinking', 'yes', 'trying to quit']
speaks (string, 7648 distinct): ['english', 'english (fluently)', 'english (fluently), spanish (poorly)', 'english (fluently), spanish (okay)', 'english (fluently), spanish (fluently)', 'english, spanish', 'english (fluently), french (poorly)', 'english, spanish (okay)', 'english, spanish (poorly)', 'english (fluently), chinese (fluently)']
status (string, 5 distinct): ['single', 'seeing someone', 'available', 'married', 'unknown']
'''

CONTEXT = "OKCupid Online Dating Profile"
TARGET = CuratedTarget(raw_name="age", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
