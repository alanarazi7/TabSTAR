from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: social_bias_frames
====
Examples: 147139
====
URL: https://www.openml.org/search?type=data&id=46701
====
Description: Warning: this document and dataset contain content that may be offensive or upsetting.

Social Bias Frames is a new way of representing the biases and offensiveness that are implied in language. For example, these frames are meant to distill the implication that "women (candidates) are less qualified" behind the statement "we shouldn't lower our standards to hire more women." The Social Bias Inference Corpus (SBIC) supports large-scale learning and evaluation of social implications with over 150k structured annotations of social media posts, spanning over 34k implications about a thousand demographic groups.

Supported Tasks and Leaderboards
This dataset supports both classification and generation. Sap et al. developed several models using the SBIC. They report an F1 score of 78.8 in predicting whether the posts in the test set were offensive, an F1 score of 78.6 in predicting whether the posts were intending to be offensive, an F1 score of 80.7 in predicting whether the posts were lewd, and an F1 score of 69.9 in predicting whether the posts were targeting a specific group.

Another of Sap et al.'s models performed better in the generation task. They report a BLUE score of 77.9, a Rouge-L score of 68.7, and a WMD score of 0.74 in generating a description of the targeted group given a post as well as a BLUE score of 52.6, a Rouge-L score of 44.9, and a WMD score of 2.79 in generating a description of the implied offensive statement given a post. See the paper for further details.

Languages
The language in SBIC is predominantly white-aligned English (78%, using a lexical dialect detector, Blodgett et al., 2016). The curators find less than 10 percentage of posts in SBIC are detected to have the AAE dialect category. The BCP-47 language tag is, presumably, en-US.

The main aim for this dataset is to cover a wide variety of social biases that are implied in text, both subtle and overt, and make the biases representative of real world discrimination that people experience RWJF 2017. The curators also included some innocuous statements, to balance out biases, offensive, or harmful content.

Source Data
The curators included online posts from the following sources sometime between 2014-2019:

r/darkJokes, r/meanJokes, r/offensiveJokes
Reddit microaggressions (Breitfeller et al., 2019)
Toxic language detection Twitter corpora (Waseem & Hovy, 2016; Davidson et al., 2017; Founa et al., 2018)
Data scraped from hate sites (Gab, Stormfront, r/incels, r/mensrights)

columns:
whoTarget: a string, '0.0' if the target is a group, '1.0' if the target is an individual, and blank if the post is not offensive
intentYN: a string indicating if the intent behind the statement was to offend. This is a categorical variable with four possible answers, '1.0' if yes, '0.66' if probably, '0.33' if probably not, and '0.0' if no.
sexYN: a string indicating whether the post contains a sexual or lewd reference. This is a categorical variable with three possible answers, '1.0' if yes, '0.5' if maybe, '0.0' if no.
sexReason: a string containing a free text explanation of what is sexual if indicated so, blank otherwise
offensiveYN (target): a string indicating if the post could be offensive to anyone. This is a categorical variable with three possible answers, '1.0' if yes, '0.5' if maybe, '0.0' if no.
annotatorGender: a string indicating the gender of the MTurk worker
annotatorMinority: a string indicating whether the MTurk worker identifies as a minority
sexPhrase: a string indicating which part of the post references something sexual, blank otherwise
speakerMinorityYN: a string indicating whether the speaker was part of the same minority group that's being targeted. This is a categorical variable with three possible answers, '1.0' if yes, '0.5' if maybe, '0.0' if no.
WorkerId: a string hashed version of the MTurk workerId
HITId: a string id that uniquely identifies each post
annotatorPolitics: a string indicating the political leaning of the MTurk worker
annotatorRace: a string indicating the race of the MTurk worker
annotatorAge: a string indicating the age of the MTurk worker
post: a string containing the text of the post that was annotated
targetMinority: a string indicating the demographic group targeted
targetCategory: a string indicating the high-level category of the demographic group(s) targeted
targetStereotype: a string containing the implied statement
dataSource: a string indicating the source of the post (t/...: means Twitter, r/...: means a subreddit)

paper_url = "https://aclanthology.org/2020.acl-main.486.pdf"

original_data_url = "https://huggingface.co/datasets/allenai/social_bias_frames"
====
Target Variable: offensiveYN (string, 3 distinct): ['1.0', '0.0', '0.5']
====
Features:

whoTarget (string, 2 distinct): ['1.0', '0.0']
intentYN (string, 4 distinct): ['1.0', '0.0', '0.66', '0.33']
sexYN (string, 3 distinct): ['0.0', '1.0', '0.5']
sexReason (string, 5238 distinct): ['sex', 'rape', 'refers to sex', 'implies sexual situations', 'oral sex', 'sex act', 'This is a sexual reference.', 'refers to a sex act', 'references sex', 'sexual intercourse']
annotatorGender (string, 5 distinct): ['woman', 'man', 'na', 'nonBinary', 'transman']
annotatorMinority (string, 82 distinct): ['none', 'women', 'None', 'female', 'pacific islander', 'veteran', 'disabled, bisexual', 'racial minority, gender', 'Hispanic, female, bisexual', 'No']
sexPhrase (string, 7051 distinct): ['rape', 'sex', 'the whole post', 'a rape victim', 'fuck', 'fucking', 'raped', 'pussy', 'having sex', 'fucked']
speakerMinorityYN (string, 3 distinct): ['0.0', '0.5', '1.0']
WorkerId (string, 307 distinct): ['-5870881806826799680', '1869048359460155317', '6347880360297734464', '4554125108380068779', '5330619308534303933', '8789702980570783632', '-4903257119943710595', '7912096326098817047', '2430897476320006833', '-5471608738051449110']
HITId (string, 45318 distinct): ['3AXFSPQOYQSOC5QQZMHF9RRGBKKJF5', '3AQN9REUTFAH9A7TUGNSVE3JCC9YD1', '30Y6N4AHYPQ8C9V7GLVYNIAMC2BDRO', '3U18MJKL1UGDSJX1NIJ6USNUQ6FCN6', '3DWNFENNE3PEEPT1F5LWINUPMUB4J2', '3J6BHNX0U9MVBH9PFV04IB02S5FKNV', '3G9UA71JVVOBXCBKH3J3LZEZGBJ7J4', '3XEIP58NL0H0CC5Q3JNZSYKQRN9ZLJ', '37VUR2VJ6AJAECQCIVL8XLARKQSC1J', '3S1WOPCJFGNW6RQU4BX9R96GZMSJEG']
annotatorPolitics (string, 7 distinct): ['liberal', 'mod-liberal', 'mod-cons', 'other', 'cons', 'libert', 'na']
annotatorRace (string, 7 distinct): ['white', 'hisp', 'asian', 'black', 'other', 'na', 'native']
annotatorAge (string, 51 distinct): ['35.0', '30.0', '25.0', '32.0', '38.0', '23.0', '29.0', '27.0', '49.0', '34.0']
post (string, 44671 distinct): ['why are black people better than jews at basketball ? because the kikes wo nt drop the dimes and the coons are great at stealing .', 'A gay guy, A black guy, and A Jew get hit by a bus They all appear in front of the gates of heaven. The Angel says. "Welcome to Heaven!" The three guys beg him for a second chance. The angel says,"I can only do it if you promised to give up what you love doing the most." The three men immediately agreed to do that.\n\nSuddenly, they\'re on the street and the bus is passing by. They cautiously cross the street, and continue walking, until they come across a car which has an open window and a purse. The gay guy says to the black guy, "Don\'t do it man, you\'re stronger than that!" But he reaches in anyway. Poof! Black guy disappears.\n\nThey continue walking until the Jew comes across a penny. He bends down with the gay guy behind him. Poof! Gay guy disappears.', 'Education system is messed up for the same reason everything else is messed up - Jews , non-whites and liberal women run and ruin everything .', 'Where do you stick a black jew In the back of the oven', 'racism&#128564; isnt real&#128581;&#127995;\u200d♂️ anymore, you stupid niggers&#128514;&#128514;&#128514;', "Things I'm sick of today: \nInterracial couples on TV\nFat models that I'm being told are beautiful\nFags\n", 'Some bad ass females end up with a nigga that is barley attractive cause all you good looking niggas are hoes and worse then bitches ☺️', 'RT @13ReasonsFans: this fucked me up https://t.co/58gm0yEn6Y', 'what is the difference between a feminist and a suicide vest ? i can turn on a suicide vest .', 'I ❤️ watching niggers, spics and broke liberal college faggits who pay no income tax throw a huge tantrum becuz White people who pay income tax will get to keep a tiny little bit more of their own 💰']
targetMinority (string, 1413 distinct): ['black folks', 'women', 'jewish folks', 'muslim folks', 'gay men', 'asian folks', 'physically disabled folks', 'liberals', 'assault victims', 'latino/latina folks']
targetCategory (string, 7 distinct): ['race', 'gender', 'culture', 'victim', 'disabled', 'social', 'body']
targetStereotype (string, 32026 distinct): ['trivializes harm to victims', 'black people are criminals', 'trivializes harm to victims.', 'women are sex objects', 'muslims are terrorists', 'women are bitches', 'black folks are criminals', 'black people are inferior', 'black folks are inferior', 'black people are worthless']
dataSource (string, 12 distinct): ['r/darkjokes', 't/founta', 'Stormfront', 'r/meanjokes', 'Gab', 'r/Incels', 't/davidson', 'redditMicroagressions', 't/waseem', 'r/MensRights']
'''

CONTEXT = "Social Bias Frames"
TARGET = CuratedTarget(raw_name="offensiveYN", new_name="Offensive Level",
                       task_type=SupervisedTask.MULTICLASS, label_mapping={'0.0': 'No', '0.5': 'Maybe', '1.0': "Yes"})
COLS_TO_DROP = ["HITId", "WorkerId",]
FEATURES = []