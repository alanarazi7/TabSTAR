from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: funpedia
====
Examples: 29819
====
URL: https://www.openml.org/search?type=data&id=46692
====
Description: Funpedia- Funpedia (Miller et al., 2017) contains rephrased Wikipedia sentences in a more conversational way. The curators retained only biography related sentences and annotate similar to Wikipedia, to give ABOUT labels.

The Multi-Dimensional Gender Bias Classification dataset is based on a general framework that decomposes gender bias in text along several pragmatic and semantic dimensions: bias from the gender of the person being spoken about, bias from the gender of the person being spoken to, and bias from the gender of the speaker. It contains seven large scale datasets automatically annotated for gender information (there are eight in the original project but the Wikipedia set is not included in the HuggingFace distribution), one crowdsourced evaluation benchmark of utterance-level gender rewrites, a list of gendered names, and a list of gendered words in English.
text-classification-other-gender-bias: The dataset can be used to train a model for classification of various kinds of gender bias. The model performance is evaluated based on the accuracy of the predicted labels as compared to the given labels in the dataset. Dinan et al's (2020) Transformer model achieved an average of 67.13 accuracy in binary gender prediction across the ABOUT, TO, and AS tasks. 

This is the dataset 'funpedia', it description is as follows:
text: the text to be classified.
gender(target): a classification label, with possible values including gender-neutral (0), female (1), male (2), indicating the gender of the person being talked about.
persona: a string describing the persona assigned to the user when talking about the entity.
title: a string naming the entity the text is about.

paper_url = "https://arxiv.org/pdf/1509.01626"

original_data_url = "https://huggingface.co/datasets/facebook/md_gender_bias/tree/10c34c50ef78b4a42f6d4eeac80a0ef2d190cd07/funpedia"
====
Target Variable: gender (numeric, 3 distinct): ['2', '1', '0']
====
Features:

text (string, 29818 distinct): ['Michael Richard Adams began with managerial career as player-manager for Fulham in 1996 .is an English former professional footballer and football manager.', 'Max Landis is a comic book writer who wrote Chronicle, American Ultra, and Victor Frankestein.', 'Bun Sha Pai is a peaceful uninhabited island in the Tai Po District of Hong Kong.', 'What an incredible accomplishment! Truly remarkable!', 'I think Yu Dafu is such a great Chinese short story writer and poet!', 'Plaza Rajah Sulayman, is a public square within manila, its often populated and has great food', 'What an incredible family history. Leukemia is so sad however, what a shame.', 'I love the spirit and dedication. A true testament to the hard work and effort.', 'Kino Delorge is a great football player, currently with FC Dordrecht on loan from Genk.', 'John Bruce Young was a jack of all trades, New Zealand baker, policeman, unionist and police commissioner and lived 1888–1952.']
title (string, 29239 distinct): ['Vampire Clan', 'Dreamland Bar-B-Que', 'Richard Pootmans', 'Lorna Webb', 'Stitt House', 'Elvet Jones', 'Ezra Chadza', 'Jimmy Britt', 'Lewis Mountain', 'William Pinch']
persona (string, 64 distinct): ['Respectful', 'Kind', 'Empathetic', 'Compassionate (Sympathetic, Warm)', 'Sweet', 'Warm', 'Charming', 'Caring', 'Happy', 'Confident']
'''

CONTEXT = "Gender Bias Classification - Funpedia"
TARGET = CuratedTarget(raw_name="gender", new_name="Gender being talked about",
                       task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"0": "Gender Neutral", "1": "Female", "2": "Male"})
COLS_TO_DROP = []
FEATURES = []