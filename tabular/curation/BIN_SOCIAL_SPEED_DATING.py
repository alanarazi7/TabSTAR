from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: SpeedDating
====
URL: https://www.openml.org/search?type=data&id=40536
====
Description: **Author**: Ray Fisman and Sheena Iyengar  
**Source**: [Columbia Business School](http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/) - 2004  
**Please cite**: None  

This data was gathered from participants in experimental speed dating events from 2002-2004. During the events, the attendees would have a four-minute "first date" with every other participant of the opposite sex. At the end of their four minutes, participants were asked if they would like to see their date again. They were also asked to rate their date on six attributes: Attractiveness, Sincerity, Intelligence, Fun, Ambition, and Shared Interests. The dataset also includes questionnaire data gathered from participants at different points in the process. These fields include: demographics, dating habits, self-perception across key attributes, beliefs on what others find valuable in a mate, and lifestyle information. 

### Attribute Information
```
 * gender: Gender of self  
 * age: Age of self  
 * age_o: Age of partner  
 * d_age: Difference in age  
 * race: Race of self  
 * race_o: Race of partner  
 * samerace: Whether the two persons have the same race or not.  
 * importance_same_race: How important is it that partner is of same race?  
 * importance_same_religion: How important is it that partner has same religion?  
 * field: Field of study  
 * pref_o_attractive: How important does partner rate attractiveness  
 * pref_o_sinsere: How important does partner rate sincerity  
 * pref_o_intelligence: How important does partner rate intelligence  
 * pref_o_funny: How important does partner rate being funny  
 * pref_o_ambitious: How important does partner rate ambition  
 * pref_o_shared_interests: How important does partner rate having shared interests  
 * attractive_o: Rating by partner (about me) at night of event on attractiveness  
 * sincere_o: Rating by partner (about me) at night of event on sincerity  
 * intelligence_o: Rating by partner (about me) at night of event on intelligence  
 * funny_o: Rating by partner (about me) at night of event on being funny  
 * ambitous_o: Rating by partner (about me) at night of event on being ambitious  
 * shared_interests_o: Rating by partner (about me) at night of event on shared interest  
 * attractive_important: What do you look for in a partner - attractiveness  
 * sincere_important: What do you look for in a partner - sincerity  
 * intellicence_important: What do you look for in a partner - intelligence  
 * funny_important: What do you look for in a partner - being funny  
 * ambtition_important: What do you look for in a partner - ambition  
 * shared_interests_important: What do you look for in a partner - shared interests  
 * attractive: Rate yourself - attractiveness  
 * sincere: Rate yourself - sincerity   
 * intelligence: Rate yourself - intelligence   
 * funny: Rate yourself - being funny   
 * ambition: Rate yourself - ambition  
 * attractive_partner: Rate your partner - attractiveness  
 * sincere_partner: Rate your partner - sincerity   
 * intelligence_partner: Rate your partner - intelligence   
 * funny_partner: Rate your partner - being funny   
 * ambition_partner: Rate your partner - ambition   
 * shared_interests_partner: Rate your partner - shared interests  
 * sports: Your own interests [1-10]  
 * tvsports  
 * exercise  
 * dining  
 * museums  
 * art  
 * hiking  
 * gaming  
 * clubbing  
 * reading  
 * tv  
 * theater  
 * movies  
 * concerts  
 * music  
 * shopping  
 * yoga  
 * interests_correlate: Correlation between participant’s and partner’s ratings of interests.  
 * expected_happy_with_sd_people: How happy do you expect to be with the people you meet during the speed-dating event?  
 * expected_num_interested_in_me: Out of the 20 people you will meet, how many do you expect will be interested in dating you?  
 * expected_num_matches: How many matches do you expect to get?  
 * like: Did you like your partner?  
 * guess_prob_liked: How likely do you think it is that your partner likes you?   
 * met: Have you met your partner before?  
 * decision: Decision at night of event.
 * decision_o: Decision of partner at night of event.  
 * match: Match (yes/no)
```

### Relevant paper

Raymond Fisman; Sheena S. Iyengar; Emir Kamenica; Itamar Simonson.   
Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment.   
The Quarterly Journal of Economics, Volume 121, Issue 2, 1 May 2006, Pages 673–697,   
[https://doi.org/10.1162/qjec.2006.121.2.673](https://doi.org/10.1162/qjec.2006.121.2.673)
====
Target Variable: match (nominal, 2 distinct): ['0', '1']
====
Features:

has_null (nominal, 2 distinct): ['1', '0']
wave (numeric, 21 distinct): ['21', '11', '9', '14', '15', '4', '2', '7', '19', '12']
gender (nominal, 2 distinct): ['male', 'female']
age (numeric, 119 distinct): ['27.0', '23.0', '26.0', '24.0', '25.0', '28.0', '22.0', '29.0', '30.0', '21.0']
age_o (numeric, 128 distinct): ['27.0', '23.0', '26.0', '24.0', '25.0', '28.0', '22.0', '29.0', '30.0', '21.0']
d_age (numeric, 35 distinct): ['1', '2', '3', '4', '5', '0', '6', '7', '8', '9']
d_d_age (nominal, 4 distinct): ['[2-3]', '[4-6]', '[0-1]', '[7-37]']
race (nominal, 6 distinct): ['European/Caucasian-American', 'Asian/Pacific Islander/Asian-American', 'Latino/Hispanic American', 'Other', 'Black/African American']
race_o (nominal, 6 distinct): ['European/Caucasian-American', 'Asian/Pacific Islander/Asian-American', 'Latino/Hispanic American', 'Other', 'Black/African American']
samerace (nominal, 2 distinct): ['0', '1']
importance_same_race (numeric, 90 distinct): ['1.0', '3.0', '2.0', '8.0', '5.0', '7.0', '6.0', '4.0', '9.0', '10.0']
importance_same_religion (numeric, 89 distinct): ['1.0', '3.0', '2.0', '5.0', '6.0', '4.0', '8.0', '7.0', '10.0', '9.0']
d_importance_same_race (nominal, 3 distinct): ['[2-5]', '[0-1]', '[6-10]']
d_importance_same_religion (nominal, 3 distinct): ['[0-1]', '[2-5]', '[6-10]']
field (nominal, 260 distinct): ['Business', 'MBA', 'Law', 'Social Work', 'International Affairs', 'Electrical Engineering', 'Psychology', 'law', 'Finance', 'business']
pref_o_attractive (numeric, 183 distinct): ['20.0', '15.0', '25.0', '10.0', '30.0', '40.0', '50.0', '35.0', '16.0', '15.38']
pref_o_sincere (numeric, 167 distinct): ['20.0', '10.0', '15.0', '25.0', '30.0', '5.0', '18.0', '0.0', '16.0', '16.67']
pref_o_intelligence (numeric, 154 distinct): ['20.0', '25.0', '30.0', '15.0', '10.0', '18.0', '19.0', '35.0', '16.0', '19.23']
pref_o_funny (numeric, 169 distinct): ['20.0', '15.0', '10.0', '25.0', '30.0', '5.0', '18.0', '16.0', '17.0', '19.23']
pref_o_ambitious (numeric, 189 distinct): ['10.0', '15.0', '5.0', '0.0', '20.0', '16.0', '18.0', '14.0', '8.0', '12.0']
pref_o_shared_interests (numeric, 214 distinct): ['10.0', '15.0', '20.0', '5.0', '0.0', '16.0', '18.0', '12.0', '14.0', '8.0']
d_pref_o_attractive (nominal, 3 distinct): ['[21-100]', '[16-20]', '[0-15]']
d_pref_o_sincere (nominal, 3 distinct): ['[16-20]', '[0-15]', '[21-100]']
d_pref_o_intelligence (nominal, 3 distinct): ['[16-20]', '[21-100]', '[0-15]']
d_pref_o_funny (nominal, 3 distinct): ['[16-20]', '[0-15]', '[21-100]']
d_pref_o_ambitious (nominal, 3 distinct): ['[0-15]', '[16-20]', '[21-100]']
d_pref_o_shared_interests (nominal, 3 distinct): ['[0-15]', '[16-20]', '[21-100]']
attractive_o (numeric, 230 distinct): ['6.0', '7.0', '5.0', '8.0', '4.0', '9.0', '3.0', '10.0', '2.0', '1.0']
sinsere_o (numeric, 301 distinct): ['8.0', '7.0', '6.0', '9.0', '10.0', '5.0', '4.0', '3.0', '2.0', '1.0']
intelligence_o (numeric, 323 distinct): ['8.0', '7.0', '6.0', '9.0', '10.0', '5.0', '4.0', '3.0', '2.0', '1.0']
funny_o (numeric, 377 distinct): ['7.0', '6.0', '8.0', '5.0', '4.0', '9.0', '10.0', '3.0', '2.0', '1.0']
ambitous_o (numeric, 737 distinct): ['7.0', '8.0', '6.0', '5.0', '9.0', '10.0', '4.0', '3.0', '2.0', '1.0']
shared_interests_o (numeric, 1091 distinct): ['5.0', '6.0', '7.0', '4.0', '8.0', '3.0', '2.0', '9.0', '1.0', '10.0']
d_attractive_o (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_sinsere_o (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_intelligence_o (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_funny_o (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_ambitous_o (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_shared_interests_o (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
attractive_important (numeric, 173 distinct): ['20.0', '15.0', '25.0', '10.0', '30.0', '40.0', '50.0', '35.0', '16.0', '15.38']
sincere_important (numeric, 157 distinct): ['20.0', '10.0', '15.0', '25.0', '30.0', '5.0', '18.0', '0.0', '16.0', '16.67']
intellicence_important (numeric, 144 distinct): ['20.0', '25.0', '30.0', '15.0', '10.0', '18.0', '35.0', '19.0', '16.0', '19.23']
funny_important (numeric, 160 distinct): ['20.0', '15.0', '10.0', '25.0', '30.0', '5.0', '18.0', '16.0', '17.0', '19.23']
ambtition_important (numeric, 181 distinct): ['10.0', '15.0', '5.0', '0.0', '20.0', '16.0', '18.0', '14.0', '8.0', '12.0']
shared_interests_important (numeric, 206 distinct): ['10.0', '15.0', '20.0', '5.0', '0.0', '16.0', '18.0', '12.0', '14.0', '8.0']
d_attractive_important (nominal, 3 distinct): ['[21-100]', '[16-20]', '[0-15]']
d_sincere_important (nominal, 3 distinct): ['[16-20]', '[0-15]', '[21-100]']
d_intellicence_important (nominal, 3 distinct): ['[16-20]', '[21-100]', '[0-15]']
d_funny_important (nominal, 3 distinct): ['[16-20]', '[0-15]', '[21-100]']
d_ambtition_important (nominal, 3 distinct): ['[0-15]', '[16-20]', '[21-100]']
d_shared_interests_important (nominal, 3 distinct): ['[0-15]', '[16-20]', '[21-100]']
attractive (numeric, 114 distinct): ['7.0', '8.0', '6.0', '9.0', '5.0', '10.0', '4.0', '3.0', '2.0']
sincere (numeric, 114 distinct): ['9.0', '8.0', '10.0', '7.0', '6.0', '5.0', '4.0', '2.0', '3.0']
intelligence (numeric, 114 distinct): ['8.0', '9.0', '7.0', '6.0', '10.0', '5.0', '3.0', '4.0', '2.0']
funny (numeric, 113 distinct): ['8.0', '9.0', '10.0', '7.0', '6.0', '5.0', '4.0', '3.0']
ambition (numeric, 114 distinct): ['8.0', '7.0', '9.0', '10.0', '6.0', '5.0', '4.0', '3.0', '2.0']
d_attractive (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_sincere (nominal, 3 distinct): ['[9-10]', '[6-8]', '[0-5]']
d_intelligence (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_funny (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_ambition (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
attractive_partner (numeric, 219 distinct): ['6.0', '7.0', '5.0', '8.0', '4.0', '9.0', '3.0', '10.0', '2.0', '1.0']
sincere_partner (numeric, 291 distinct): ['8.0', '7.0', '6.0', '9.0', '10.0', '5.0', '4.0', '3.0', '2.0', '1.0']
intelligence_partner (numeric, 313 distinct): ['8.0', '7.0', '6.0', '9.0', '10.0', '5.0', '4.0', '3.0', '2.0', '1.0']
funny_partner (numeric, 366 distinct): ['7.0', '6.0', '8.0', '5.0', '4.0', '9.0', '10.0', '3.0', '2.0', '1.0']
ambition_partner (numeric, 727 distinct): ['7.0', '8.0', '6.0', '5.0', '9.0', '10.0', '4.0', '3.0', '2.0', '1.0']
shared_interests_partner (numeric, 1082 distinct): ['5.0', '6.0', '7.0', '4.0', '8.0', '3.0', '2.0', '9.0', '1.0', '10.0']
d_attractive_partner (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_sincere_partner (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_intelligence_partner (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_funny_partner (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_ambition_partner (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_shared_interests_partner (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
sports (numeric, 89 distinct): ['8.0', '7.0', '10.0', '9.0', '5.0', '6.0', '3.0', '4.0', '2.0', '1.0']
tvsports (numeric, 89 distinct): ['1.0', '2.0', '7.0', '3.0', '5.0', '8.0', '4.0', '6.0', '9.0', '10.0']
exercise (numeric, 89 distinct): ['8.0', '7.0', '6.0', '5.0', '9.0', '10.0', '4.0', '3.0', '2.0', '1.0']
dining (numeric, 89 distinct): ['8.0', '9.0', '10.0', '7.0', '6.0', '5.0', '4.0', '3.0', '2.0', '1.0']
museums (numeric, 90 distinct): ['7.0', '8.0', '9.0', '6.0', '10.0', '5.0', '4.0', '3.0', '2.0', '1.0']
art (numeric, 90 distinct): ['8.0', '7.0', '5.0', '10.0', '6.0', '9.0', '3.0', '4.0', '2.0', '1.0']
hiking (numeric, 90 distinct): ['8.0', '7.0', '6.0', '3.0', '5.0', '9.0', '4.0', '2.0', '10.0', '1.0']
gaming (numeric, 91 distinct): ['1.0', '2.0', '3.0', '5.0', '6.0', '7.0', '4.0', '8.0', '9.0', '14.0']
clubbing (numeric, 90 distinct): ['8.0', '7.0', '6.0', '9.0', '5.0', '1.0', '3.0', '4.0', '2.0', '10.0']
reading (numeric, 90 distinct): ['9.0', '8.0', '10.0', '7.0', '6.0', '5.0', '3.0', '4.0', '2.0', '13.0']
tv (numeric, 89 distinct): ['6.0', '5.0', '7.0', '8.0', '1.0', '4.0', '2.0', '3.0', '9.0', '10.0']
theater (numeric, 90 distinct): ['7.0', '8.0', '9.0', '5.0', '6.0', '10.0', '4.0', '3.0', '2.0', '1.0']
movies (numeric, 89 distinct): ['8.0', '9.0', '7.0', '10.0', '6.0', '5.0', '4.0', '3.0', '2.0', '0.0']
concerts (numeric, 90 distinct): ['7.0', '8.0', '6.0', '9.0', '5.0', '10.0', '4.0', '3.0', '2.0', '1.0']
music (numeric, 89 distinct): ['10.0', '8.0', '9.0', '7.0', '6.0', '5.0', '4.0', '3.0', '1.0', '2.0']
shopping (numeric, 89 distinct): ['7.0', '5.0', '6.0', '2.0', '8.0', '9.0', '4.0', '3.0', '10.0', '1.0']
yoga (numeric, 90 distinct): ['1.0', '2.0', '3.0', '7.0', '6.0', '5.0', '4.0', '8.0', '9.0', '10.0']
d_sports (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_tvsports (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
d_exercise (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_dining (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_museums (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_art (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_hiking (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
d_gaming (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
d_clubbing (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_reading (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_tv (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
d_theater (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_movies (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_concerts (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_music (nominal, 3 distinct): ['[6-8]', '[9-10]', '[0-5]']
d_shopping (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
d_yoga (nominal, 3 distinct): ['[0-5]', '[6-8]', '[9-10]']
interests_correlate (numeric, 313 distinct): ['0.31', '0.13', '0.24', '0.19', '0.11', '0.43', '0.09', '0.32', '0.34', '0.27']
d_interests_correlate (nominal, 3 distinct): ['[0-0.33]', '[0.33-1]', '[-1-0]']
expected_happy_with_sd_people (numeric, 111 distinct): ['5.0', '6.0', '7.0', '4.0', '3.0', '8.0', '2.0', '9.0', '10.0', '1.0']
expected_num_interested_in_me (numeric, 6596 distinct): ['3.0', '2.0', '5.0', '4.0', '10.0', '1.0', '0.0', '6.0', '8.0', '20.0']
expected_num_matches (numeric, 1190 distinct): ['2.0', '3.0', '1.0', '4.0', '5.0', '0.0', '6.0', '8.0', '7.0', '10.0']
d_expected_happy_with_sd_people (nominal, 3 distinct): ['[5-6]', '[7-10]', '[0-4]']
d_expected_num_interested_in_me (nominal, 3 distinct): ['[0-3]', '[4-9]', '[10-20]']
d_expected_num_matches (nominal, 3 distinct): ['[0-2]', '[3-5]', '[5-18]']
like (numeric, 258 distinct): ['7.0', '6.0', '5.0', '8.0', '4.0', '9.0', '3.0', '2.0', '10.0', '1.0']
guess_prob_liked (numeric, 328 distinct): ['5.0', '6.0', '7.0', '4.0', '3.0', '8.0', '2.0', '1.0', '9.0', '10.0']
d_like (nominal, 3 distinct): ['[6-8]', '[0-5]', '[9-10]']
d_guess_prob_liked (nominal, 3 distinct): ['[5-6]', '[0-4]', '[7-10]']
met (numeric, 382 distinct): ['0.0', '1.0', '7.0', '5.0', '3.0', '8.0', '6.0']
'''

CONTEXT = "Speed Dating Matching"
TARGET = CuratedTarget(raw_name="match", new_name="Successful Match", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
