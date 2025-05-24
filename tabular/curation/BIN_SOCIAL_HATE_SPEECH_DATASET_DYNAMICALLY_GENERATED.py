from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Dynamically-Generated-Hate-Speech-Dataset
====
Examples: 41144
====
URL: https://www.openml.org/search?type=data&id=46683
====
Description: The Dynamically Generated Hate Speech Dataset is provided in one table.

'acl.id' is the unique ID of the entry.

'Text' is the content which has been entered. All content is synthetic.

'Label' is a binary variable, indicating whether or not the content has been identified as hateful. It takes two values: hate, nothate.

'Type' is a categorical variable, providing a secondary label for hateful content. For hate it can take five values: Animosity, Derogation, Dehumanization, Threatening and Support for Hateful Entities. Please see the paper for more detail. For nothate the 'type' is 'none'. In round 1 the 'type' was not given and is marked as 'notgiven'.

'Target' is a categorical variable, providing the group that is attacked by the hate. It can include intersectional characteristics and multiple groups can be identified. For nothate the type is 'none'. Note that in round 1 the 'target' was not given and is marked as 'notgiven'.

'Level' reports whether the entry is original content or a perturbation.

'Round' is a categorical variable. It gives the round of data entry (1, 2, 3 or 4) with a letter for whether the entry is original content ('a') or a perturbation ('b'). Perturbations were not made for round 1.

'Round.base' is a categorical variable. It gives the round of data entry, indicated with just a number (1, 2, 3 or 4).

'Split' is a categorical variable. it gives the data split that the entry has been assigned to. This can take the values 'train', 'dev' and 'test'. The choice of splits is explained in the paper.

'Annotator' is a categorical variable. It gives the annotator who entered the content. Annotator IDs are random alphanumeric strings. There are 20 annotators in the dataset.

'acl.id.matched' is the ID of the matched entry, connecting the original (given in 'acl.id') and the perturbed version.

paper_url = "https://aclanthology.org/2021.acl-long.132.pdf"
original_data_url = "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset"
====
Target Variable: label (string, 2 distinct): ['hate', 'nothate']
====
Features:

extra.id (numeric, 41144 distinct): ['1', '27565', '27507', '27508', '27509', '27510', '27511', '27512', '27513', '27514']
acl.id (string, 41144 distinct): ['acl1', 'acl34806', 'acl34754', 'acl34755', 'acl34756', 'acl34757', 'acl34758', 'acl34759', 'acl3476', 'acl34760']
X1 (numeric, 41144 distinct): ['1', '27565', '27507', '27508', '27509', '27510', '27511', '27512', '27513', '27514']
text (string, 41134 distinct): ['Blacks are such pondlife', 'Muslims are a disgrace', 'Politicians are scum', 'Politicians are crap', 'black convicts have a disproportionately higher rate of incarceration than other ethnicities', 'I endorse the genocide of Muslims', 'It looks like physics? Or is it black fuckery?', "We are burning the Black Lives Matter sign. We're clearly tryin to terrorize the Black community and send a message that No, black lives don't matter to us. We will never allow you to have equal rights. We will use violence to keep you down.", '"Thanks for being there for me when nobody else was around, my nigga. you\'re my brother and best friend. i hope you are doing good. rest in peace my nigga, have fun in heaven and take care." - why can the coons say nigga but not us, ffs', 'asylum seekers should be buried alive']
type (string, 7 distinct): ['none', 'derogation', 'notgiven', 'animosity', 'dehumanization', 'threatening', 'support']
target (string, 409 distinct): ['none', 'notgiven', 'wom', 'bla', 'jew', 'mus', 'trans', 'gay', 'immig', 'dis']
level (string, 2 distinct): ['original', 'perturbation']
split (string, 3 distinct): ['train', 'test', 'dev']
round.base (numeric, 4 distinct): ['1', '4', '2', '3']
annotator (string, 20 distinct): ['CAgNlUizNm', 'TrRF46JWfP', 'GNZuCtwed3', 'LqLKTtrOmx', 'TbUBpfn6iP', 'eLGzdD8Tvb', 'QiOKkCi7F8', 'vDe7GN0NrL', 'E3dsmnSPob', 'oemYWm1Tjg']
round (string, 7 distinct): ['1', '4a', '4b', '2b', '2a', '3a', '3b']
acl.id.matched (string, 30098 distinct): ['acl20309', 'acl31223', 'acl31233', 'acl31234', 'acl40592', 'acl40476', 'acl40192', 'acl31228', 'acl31229', 'acl40413']
'''

CONTEXT = "Dynamically Generated Hate Speech Dataset"
TARGET = CuratedTarget(raw_name="label", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["extra.id", "acl.id", "X1", "acl.id.matched"]
FEATURES = []