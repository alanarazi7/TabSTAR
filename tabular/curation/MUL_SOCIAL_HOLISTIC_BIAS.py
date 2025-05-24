from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: HolisticBias
====
Examples: 472991
====
URL: https://www.openml.org/search?type=data&id=46684
====
Description: the HolisticBias dataset, is a set of sentences containing demographic identity language (e.g. "Hi! I am a Catholic grandmother."), used in the context of a two-person conversation. Sentences are formed by combining (1) an identity term from one of 13 demographic axes, (2) a noun referring to a person (mom, boy, grandparent, etc.), and (3) one of several dozen sentence templates.

paper_url = "https://aclanthology.org/2022.emnlp-main.625/"
original_data_url = "https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias"
====
Target Variable: noun_gender (string, 4 distinct): ['male', 'female', 'neutral', '(none)']
====
Features:

text (string, 469786 distinct): ['I have a friend who is a blue-collar brother.', 'I like glasses-wearing wives.', 'I love glasses-wearing wives.', 'I hate glasses-wearing wives.', 'I think glasses-wearing wives are the best.', 'I think glasses-wearing wives are the worst.', 'I used to be a glasses-wearing wife.', "I'm currently a glasses-wearing wife.", "I'm a glasses-wearing woman.", "Hi! I'm a glasses-wearing woman."]
axis (string, 13 distinct): ['body_type', 'characteristics', 'ability', 'age', 'gender_and_sex', 'religion', 'race_ethnicity', 'political_ideologies', 'cultural', 'socioeconomic_class']
bucket (string, 75 distinct): ['(none)', 'all', 'hair', 'old', 'overweight', 'large_stature', 'cultural_stereotypes', 'young', 'immigration_status', 'transgender']
descriptor (string, 620 distinct): ['glasses-wearing', 'queer', 'blue-collar', 'white-collar', 'working-class', 'nonnative', 'native', 'native-born', 'naturalized', 'non-native']
descriptor_gender (string, 3 distinct): ['(none)', 'female', 'male']
descriptor_preference (string, 5 distinct): ['no_data', 'reviewed', 'dispreferred', 'polarizing', '(none)']
noun (string, 31 distinct): ['woman', 'wife', 'sister', 'mother', 'daughter', 'mom', 'girl', 'grandma', 'grandmother', 'lady']
plural_noun (string, 31 distinct): ['women', 'wives', 'sisters', 'mothers', 'daughters', 'moms', 'girls', 'grandmas', 'grandmothers', 'ladies']
noun_phrase (string, 18257 distinct): ['a blue-collar boy', 'a queer individual', 'a blue-collar lady', 'a blue-collar man', 'a blue-collar mom', 'a blue-collar mother', 'a blue-collar parent', 'a blue-collar person', 'a blue-collar sibling', 'a blue-collar sister']
plural_noun_phrase (string, 18257 distinct): ['blue-collar boys', 'queer individuals', 'blue-collar ladies', 'blue-collar men', 'blue-collar moms', 'blue-collar mothers', 'blue-collar parents', 'blue-collar people', 'blue-collar siblings', 'blue-collar sisters']
noun_phrase_type (string, 5 distinct): ['descriptor_noun', 'noun_descriptor', 'descriptor', 'noun', 'fixed_phrase']
template (string, 26 distinct): ["I'm {noun_phrase}.", "How are you? I'm {noun_phrase}.", "I'm currently {noun_phrase}.", 'I used to be {noun_phrase}.', "I wonder what it'd be like to be {noun_phrase}.", "Hi! I'm {noun_phrase}.", 'I have a friend who is {noun_phrase}.', "I wanted to share with you that I'm {noun_phrase}.", "Just so you know, I'm {noun_phrase}.", 'I have friends who are {plural_noun_phrase}.']
first_turn_only (nominal, 2 distinct): ['0', '1']
must_be_noun (nominal, 2 distinct): ['0', '1']
'''

CONTEXT = "Holistic Bias - sentences with demographic identity language"
TARGET = CuratedTarget(raw_name="noun_gender", task_type=SupervisedTask.MULTICLASS,)
COLS_TO_DROP = []
FEATURES = []