from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: jigsaw_unintended_bias100K
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46654
====
Description: Predict whether online social media comments are toxic based on their text
    and additional tabular features providing information about the post (e.g. likes, rating, date created, etc.). This dataset originates from a 2019 Kaggle competition
    (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
    in which the 1st place solution utilized dataset-specific tricks such as a Bucket Sequencing Collator,
    auxiliary domain-specific prediction tasks for models, and a custom mimic loss function for training.
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: target (nominal, 2 distinct): ['0', '1']
====
Features:

id (numeric, 100000 distinct): ['964172', '5891049', '5092409', '5426881', '5651990', '811090', '391562', '5847686', '888286', '5414537']
comment_text (string, 99662 distinct): ['No.', 'Well said!', 'Well said.', 'Thank you.', '.', 'Sᴛᴀʀᴛ ᴡᴏʀᴋɪɴɢ ғʀᴏᴍ ʜᴏᴍᴇ! Gʀᴇᴀᴛ ᴊᴏʙ ғᴏʀ sᴛᴜᴅᴇɴᴛs, sᴛᴀʏ-ᴀᴛ-ʜᴏᴍᴇ ᴍᴏᴍs ᴏʀ ᴀɴʏᴏɴᴇ ɴᴇᴇᴅɪɴɢ ᴀɴ ᴇxᴛʀᴀ ɪɴᴄᴏᴍᴇ... Yᴏᴜ ᴏɴʟʏ ɴᴇᴇᴅ ᴀ ᴄᴏᴍᴘᴜᴛᴇʀ ᴀɴᴅ ᴀ ʀᴇʟɪᴀʙʟᴇ ɪɴᴛᴇʀɴᴇᴛ ᴄᴏɴɴᴇᴄᴛɪᴏɴ... Mᴀᴋᴇ $90 ʜᴏᴜʀʟʏ ᴀɴᴅ ᴜᴘ ᴛᴏ $12000 ᴀ ᴍᴏɴᴛʜ ʙʏ ғᴏʟʟᴏᴡɪɴɢ ʟɪɴᴋ ᴀᴛ ᴛʜᴇ ʙᴏᴛᴛᴏᴍ ᴀɴᴅ sɪɢɴɪɴɢ ᴜᴘ... Yᴏᴜ ᴄᴀɴ ʜᴀᴠᴇ ʏᴏᴜʀ ғɪʀsᴛ ᴄʜᴇᴄᴋ ʙʏ ᴛʜᴇ ᴇɴᴅ ᴏғ ᴛʜɪs ᴡᴇᴇᴋ... \n\n+++++++++http://www.cashapp24.com/', 'Thank you!', 'What?', 'I agree.', 'Why?']
severe_toxicity (numeric, 290 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.0143', '0.0286', '0.025', '0.0125', '0.0429', '0.0152']
obscene (numeric, 742 distinct): ['0.0', '0.1', '0.2', '0.1667', '0.3', '0.4', '0.5', '0.6', '0.0143', '0.1429']
identity_attack (numeric, 603 distinct): ['0.0', '0.1', '0.2', '0.1667', '0.3', '0.4', '0.5', '0.6', '0.0143', '0.1429']
insult (numeric, 1204 distinct): ['0.0', '0.1667', '0.2', '0.3', '0.1', '0.4', '0.5', '0.6', '0.7', '0.8']
threat (numeric, 442 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.0143', '0.0125', '0.5', '0.0132']
asian (numeric, 27 distinct): ['0.0', '1.0', '0.1667', '0.2', '0.1', '0.5', '0.3', '0.4', '0.6', '0.7']
atheist (numeric, 18 distinct): ['0.0', '1.0', '0.1', '0.75', '0.8', '0.8333', '0.1667', '0.6', '0.25', '0.7']
bisexual (numeric, 18 distinct): ['0.0', '0.1', '0.2', '0.1667', '0.3', '0.4', '0.5', '0.3333', '1.0', '0.25']
black (numeric, 24 distinct): ['0.0', '1.0', '0.8', '0.8333', '0.1', '0.6', '0.5', '0.7', '0.1667', '0.2']
buddhist (numeric, 17 distinct): ['0.0', '0.1', '0.1667', '0.5', '1.0', '0.8333', '0.75', '0.6', '0.2', '0.25']
christian (numeric, 28 distinct): ['0.0', '1.0', '0.4', '0.6', '0.3', '0.5', '0.8', '0.1667', '0.2', '0.8333']
female (numeric, 34 distinct): ['0.0', '1.0', '0.8333', '0.1667', '0.8', '0.2', '0.1', '0.7', '0.3', '0.4']
heterosexual (numeric, 21 distinct): ['0.0', '0.1', '0.1667', '1.0', '0.8333', '0.5', '0.6', '0.25', '0.8', '0.75']
hindu (numeric, 18 distinct): ['0.0', '0.1667', '0.1', '1.0', '0.2', '0.8', '0.5', '0.8333', '0.25', '0.3']
homosexual_gay_or_lesbian (numeric, 24 distinct): ['0.0', '1.0', '0.8333', '0.8', '0.1', '0.6', '0.7', '0.2', '0.1667', '0.9']
intellectual_or_learning_disability (numeric, 12 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.25', '0.6', '0.5', '0.0008']
jewish (numeric, 22 distinct): ['0.0', '1.0', '0.1', '0.8', '0.8333', '0.1667', '0.7', '0.2', '0.9', '0.5']
latino (numeric, 23 distinct): ['0.0', '0.1', '0.1667', '0.4', '0.2', '0.25', '0.3', '1.0', '0.5', '0.7']
male (numeric, 37 distinct): ['0.0', '1.0', '0.1667', '0.2', '0.1', '0.8333', '0.5', '0.8', '0.7', '0.6']
muslim (numeric, 26 distinct): ['0.0', '1.0', '0.8333', '0.8', '0.5', '0.1', '0.6', '0.2', '0.4', '0.1667']
other_disability (numeric, 13 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.1429', '0.3', '0.0016', '0.0008', '0.0006', '0.4']
other_gender (numeric, 10 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.0016', '0.0011', '0.0006', '0.6']
other_race_or_ethnicity (numeric, 29 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.3', '0.4', '0.5', '0.3333', '0.6']
other_religion (numeric, 21 distinct): ['0.0', '0.1', '0.2', '0.1667', '0.25', '0.3', '0.5', '0.4', '0.3333', '0.75']
other_sexual_orientation (numeric, 15 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.3', '0.1429', '0.3333', '0.0064', '0.0197']
physical_disability (numeric, 15 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.5', '0.25', '0.0008', '0.1429']
psychiatric_or_mental_illness (numeric, 23 distinct): ['0.0', '0.1667', '1.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '0.8333']
transgender (numeric, 22 distinct): ['0.0', '0.1', '1.0', '0.2', '0.1667', '0.5', '0.3', '0.25', '0.8333', '0.6']
white (numeric, 28 distinct): ['0.0', '1.0', '0.8', '0.8333', '0.7', '0.6', '0.1', '0.5', '0.1667', '0.9']
created_date (string, 99997 distinct): ['2015-10-13 17:16:38.081524+00', '2015-09-29 17:37:25.068440+00', '2015-10-06 18:23:45.995878+00', '2017-09-24 20:04:11.379178+00', '2016-10-31 16:50:29.711104+00', '2017-06-22 08:23:15.005369+00', '2017-04-04 19:01:00.146140+00', '2017-06-16 15:28:55.285999+00', '2017-07-25 01:15:39.369564+00', '2017-01-09 01:29:42.238578+00']
publication_id (numeric, 48 distinct): ['54', '21', '102', '13', '55', '53', '22', '105', '100', '43']
parent_id (numeric, 55371 distinct): ['6054583.0', '685285.0', '6021200.0', '379918.0', '974787.0', '5023804.0', '6118707.0', '5646323.0', '5264939.0', '5630809.0']
article_id (numeric, 37373 distinct): ['352700', '351447', '383131', '97938', '327856', '390548', '381888', '351636', '165520', '163252']
rating (string, 2 distinct): ['approved', 'rejected']
funny (numeric, 32 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
wow (numeric, 8 distinct): ['0', '1', '2', '3', '4', '5', '7', '15']
sad (numeric, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
likes (numeric, 95 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
disagree (numeric, 49 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sexual_explicit (numeric, 401 distinct): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.5', '0.0125', '0.0143', '0.0135']
identity_annotator_count (numeric, 17 distinct): ['0', '4', '10', '6', '5', '7', '11', '9', '8', '1575']
toxicity_annotator_count (numeric, 139 distinct): ['4', '10', '6', '5', '70', '80', '7', '76', '74', '66']
'''

CONTEXT = "Online Social Media Comments Toxicity"
TARGET = CuratedTarget(raw_name="target", new_name="Is Toxic", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["id", "severe_toxicity", "toxicity_annotator_count", "identity_annotator_count"]
FEATURES = [CuratedFeature(raw_name="created_date", feat_type=FeatureType.DATE)]
