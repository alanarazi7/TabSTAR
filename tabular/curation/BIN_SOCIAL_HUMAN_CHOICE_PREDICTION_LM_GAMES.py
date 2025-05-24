from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: eilamshapira/human-choice-prediction-in-language-based-games/OPE_train.csv
====
Examples: 71579
====
URL: https://www.kaggle.com/eilamshapira/human-choice-prediction-in-language-based-games/OPE_train.csv
====
Description: 
Human Choice Prediction in Language-based Games
Human decisions in response to persuasive textual reviews in language-based game

This dataset contains 87,204 human decisions collected through a large-scale experiment in language-based persuasion games. The data is split into two parts:

OPE_train.csv – 71,579 decisions from human players interacting with one group of expert agents (bots).

OPE_test.csv – 15,625 decisions from interactions with a different group of bots, representing an off-policy evaluation (OPE) scenario.

🧠 The Game Setting
In each round of the game:

A bot (expert agent) is shown 7 textual reviews of a hotel (collected from Booking.com), each with a score.

The bot selects one review and sends only its textual content to the human player (DM), hiding the numerical score.

The human then decides whether to go to the hotel or stay at home, based only on the review and past interactions.

Each player played 6 games (with 6 different bots), 10 rounds each. Only players who successfully finished all games (i.e., reached a target score) are included.

📁 What’s in the Data
Each row is a single decision round and includes:

positive_part, negative_part: The textual review parts shown to the DM.

review_score, hotelGood: The true (hidden) score and binary indicator of hotel quality.

didGo: The player’s binary decision.

Game context: round number, game ID, bot strategy ID, player ID.

Dynamic features: past decisions and rewards (e.g., last_didGo, last_review_score, last_round_positive_part), enabling sequence modeling.

🧪 Research Context
This dataset was introduced and analyzed in the following paper:

Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation
Eilam Shapira, Omer Madmon, Reut Apel, Moshe Tennenholtz, Roi Reichart (2025)

The dataset was collected as part of the experiments reported in this paper, which investigates the challenge of predicting human decisions in language-based persuasion games under an off-policy evaluation (OPE) setting.

📌 If you use this dataset in your work, please cite the paper: https://arxiv.org/abs/2305.10361

====
Features:

user_id (int64, 210 distinct): ['73', '126', '184', '8', '197', '92', '45', '181', '192', '104']
strategy_id (int64, 6 distinct): ['19', '2', '59', '5', '0', '3']
gameId (int64, 379 distinct): ['0', '1', '2', '4', '3', '5', '7', '9', '6', '8']
roundNum (int64, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
user_points (int64, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
bot_points (int64, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
last_didGo (bool, 2 distinct): ['1', '0']
last_last_didGo (bool, 2 distinct): ['0', '1']
last_didWin (bool, 2 distinct): ['1', '0']
last_last_didWin (bool, 2 distinct): ['1', '0']
last_reaction_time (int64, 16660 distinct): ['-1', '161', '149', '179', '160', '177', '173', '176', '155', '153']
hotelGood (bool, 2 distinct): ['1', '0']
last_hotelGood (object, 2 distinct): ['1', '0']
last_last_hotelGood (object, 2 distinct): ['1', '0']
positive_part (object, 2608 distinct): ['Location', 'Location.', 'Nothing', 'The location', 'Location is good', 'location', 'Great flats in the centre of town, superbly styled and comfortable.', 'Very well located comfortable property.  Staff was exceedingly personable.', "This was our second stay at the Bonifacius and it did not disappoint.  Wonderful in everyway. Location is the best, breakfast delightful and extremely comfortable and spacious rooms.  Can't wait to go back.", 'Loved the relaxing, friendly environment of the beautiful house and grounds. The kids loved running around the gardens, food was excellent and Axel and Guy delightful hosts. Will be coming back!']
negative_part (object, 2099 distinct): ['Nothing', 'Nothing.', '-', 'Nothing!', 'nothing', 'All good', '/', 'Nothing not to like', 'Nothing really', 'The interior design was excellent as was the garden. Taniya and Murielle made us feel very welcome and did everything they could to make our stay special.']
last_round_positive_part (object, 2608 distinct): ['Location', 'Location.', 'Nothing', 'The location', 'Location is good', 'location', 'Great flats in the centre of town, superbly styled and comfortable.', 'Loved the relaxing, friendly environment of the beautiful house and grounds. The kids loved running around the gardens, food was excellent and Axel and Guy delightful hosts. Will be coming back!', "This was our second stay at the Bonifacius and it did not disappoint.  Wonderful in everyway. Location is the best, breakfast delightful and extremely comfortable and spacious rooms.  Can't wait to go back.", 'This was a lovely and warm place to stay']
last_round_negative_part (object, 2099 distinct): ['Nothing', 'Nothing.', '-', 'Nothing!', 'nothing', 'All good', '/', 'Nothing not to like', 'Nothing really', 'The interior design was excellent as was the garden. Taniya and Murielle made us feel very welcome and did everything they could to make our stay special.']
last_last_round_positive_part (object, 2608 distinct): ['Location', 'Location.', 'Nothing', 'The location', 'Location is good', 'location', 'Great flats in the centre of town, superbly styled and comfortable.', "This was our second stay at the Bonifacius and it did not disappoint.  Wonderful in everyway. Location is the best, breakfast delightful and extremely comfortable and spacious rooms.  Can't wait to go back.", 'Loved the relaxing, friendly environment of the beautiful house and grounds. The kids loved running around the gardens, food was excellent and Axel and Guy delightful hosts. Will be coming back!', 'This was a lovely and warm place to stay']
last_last_round_negative_part (object, 2099 distinct): ['Nothing', 'Nothing.', '-', 'Nothing!', 'nothing', 'All good', '/', 'Nothing really', 'Nothing not to like', 'The interior design was excellent as was the garden. Taniya and Murielle made us feel very welcome and did everything they could to make our stay special.']
review_score (float64, 32 distinct): ['10.0', '9.6', '8.0', '9.0', '9.2', '7.9', '8.8', '8.3', '7.5', '7.0']
last_review_score (float64, 32 distinct): ['10.0', '9.6', '8.0', '9.0', '9.2', '7.9', '8.8', '8.3', '7.5', '7.0']
last_last_review_score (float64, 32 distinct): ['10.0', '9.6', '8.0', '9.0', '9.2', '7.9', '8.8', '8.3', '7.5', '7.0']
didGo (bool, 2 distinct): ['1', '0']
'''

CONTEXT = "Human Choice Prediction in Language-based Games"
TARGET = CuratedTarget(raw_name="didGo", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["user_id", "gameId", "strategy_id"]
FEATURES = []

DESCRIPTION = '''
Human Choice Prediction in Language-based Games
Human decisions in response to persuasive textual reviews in language-based game

This dataset contains 87,204 human decisions collected through a large-scale experiment in language-based persuasion games. The data is split into two parts:

OPE_train.csv – 71,579 decisions from human players interacting with one group of expert agents (bots).

OPE_test.csv – 15,625 decisions from interactions with a different group of bots, representing an off-policy evaluation (OPE) scenario.

🧠 The Game Setting
In each round of the game:

A bot (expert agent) is shown 7 textual reviews of a hotel (collected from Booking.com), each with a score.

The bot selects one review and sends only its textual content to the human player (DM), hiding the numerical score.

The human then decides whether to go to the hotel or stay at home, based only on the review and past interactions.

Each player played 6 games (with 6 different bots), 10 rounds each. Only players who successfully finished all games (i.e., reached a target score) are included.

📁 What’s in the Data
Each row is a single decision round and includes:

positive_part, negative_part: The textual review parts shown to the DM.

review_score, hotelGood: The true (hidden) score and binary indicator of hotel quality.

didGo: The player’s binary decision.

Game context: round number, game ID, bot strategy ID, player ID.

Dynamic features: past decisions and rewards (e.g., last_didGo, last_review_score, last_round_positive_part), enabling sequence modeling.

🧪 Research Context
This dataset was introduced and analyzed in the following paper:

Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation
Eilam Shapira, Omer Madmon, Reut Apel, Moshe Tennenholtz, Roi Reichart (2025)

The dataset was collected as part of the experiments reported in this paper, which investigates the challenge of predicting human decisions in language-based persuasion games under an off-policy evaluation (OPE) setting.

📌 If you use this dataset in your work, please cite the paper: https://arxiv.org/abs/2305.10361
'''