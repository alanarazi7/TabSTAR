from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: REASONER
====
Examples: 58497
====
URL: https://www.openml.org/search?type=data&id=46681
====
Description: REASONER is an explainable recommendation dataset. It contains the ground truths for multiple explanation purposes, for example, enhancing the recommendation persuasiveness, informativeness and satisfaction. In this dataset, the ground truth annotators are exactly the people who produce the user-item interactions, and they can make selections from the explanation candidates with multi-modalities. This dataset can be widely used for explainable recommendation, unbiased recommendation, psychology-informed recommendation and so on. Please see our paper for more details.
The dataset contains the following files.

 REASONER-Dataset
      interaction.csv
      user.csv
      video.csv
      bigfive.csv 
      tag_map.csv 
      video_map.csv 
      preview

We joined them in the next way (images and video not included):

interaction and users joined by the column "user_id"

The resulting dataset joined with video_df by the column "video_id"

The resulting dataset joined with bigfive by the column "user_id"

In addition, we converted the tags in final_df from a list of tag IDs to a string of tag contents using the tag_map dataframe
Finally, we delete the column "rating", since it can make things super easy for any classifier to predict the column "like" (target)

paper_url = "https://papers.nips.cc/paper_files/paper/2023/file/2ebf43d20e5933ab6d98225bbb908ade-Paper-Datasets_and_Benchmarks.pdf"
original_data_url = "https://reasoner2023.github.io/docs/dataset"

We limited the columns name to less than 64 characters and we ensure that all the columns are unique without non-ASCII characters.
====
Target Variable: like (numeric, 2 distinct): ['1', '0']
====
Features:

user_id (numeric, 2997 distinct): ['1139', '1148', '1140', '1128', '232', '231', '1146', '2258', '1141', '1147']
video_id (numeric, 4672 distinct): ['1680', '3986', '3836', '138', '2881', '3937', '4257', '3652', '2537', '402']
reason_tag (string, 56844 distinct): ['[5691, 5953, 5074]', '[3835, 5953, 5074]', '[0, 1, 2]', '[5074, 5953, 5691]', '[5, 6, 1203]', '[5691, 5074, 5953]', '[2958, 5074, 1688]', '[3835, 5074, 5953]', '[5953, 5691, 5074]', '[5953, 5074, 5691]']
review (string, 58279 distinct): ["I don't like this kind of video very much. I feel it is boring and boring.", "Like this kind of funny video, brain hole is also very big, read people's mood is good.", 'Ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha', 'The ghost livestock video is very funny and different classic clips blend together very harmoniously.', 'Animation is very humorous, funny fragments have a lot of, very in line with the preferences of young people now.', "I don't like watching cartoons very much. I feel it's boring.", "I don't like ghost and animal videos. I'm not interested in this style of video. I always feel that it's not very polite.", "I don't like this kind of video with animation effect very much. I feel it is not interesting.", "I don't like this kind of video very much. I feel it is boring and meaningless.", "I don't like watching this kind of video very much. I feel it is meaningless and boring."]
video_tag (string, 57772 distinct): ['[5691, 5953, 5074]', '[5953, 5074, 5691]', '[5691, 5074, 5953]', '[5074, 5691, 5953]', '[3835, 5074, 5953]', '[3835, 5953, 5074]', '[5074, 5953, 5691]', '[5953, 5691, 5074]', '[5074, 1688, 2958]', '[5953, 3835, 5074]']
interest_tag (string, 57526 distinct): ['[5691, 5953, 5074]', '[5074, 5691, 5953]', '[5953, 5691, 5074]', '[5691, 5074, 5953]', '[5074, 5953, 5691]', '[3835, 5953, 5074]', '[5953, 5074, 5691]', '[2236, 3205]', '[3205, 2236, 3687]', '[3835, 5074, 5953]']
watch_again (numeric, 2 distinct): ['0', '1']
age (numeric, 9 distinct): ['1', '2', '3', '4', '5', '6', '0', '8', '7']
gender (numeric, 2 distinct): ['1', '0']
education (numeric, 8 distinct): ['4', '3', '2', '7', '5', '1', '6', '0']
career (numeric, 21 distinct): ['20', '0', '5', '2', '3', '1', '4', '18', '10', '7']
income (numeric, 5 distinct): ['0', '1', '2', '4', '3']
address (numeric, 31 distinct): ['6', '16', '3', '7', '8', '4', '1', '2', '10', '5']
hobby (string, 670 distinct): ['Sing', 'basketball', 'Reading', 'Listen to music', 'None', 'play basketball', 'movement', 'Painting', 'Music', 'Badminton']
title (string, 4669 distinct): ['Do you want to dance too?', 'Suggestion: recite the whole text', "Mantis: Don 't kill the poodle!", 'Preface to Lanting', 'National situation', 'Kill my horse mother', 'Hua Chunying domineering "six companies," refuted American politicians \'remarks "shameless"!', 'Vegetarian Virgin Bravely broke into the farm to save the rabbit and was killed by the rabbit. The cause of the injury was thrown to the human being.', 'The beverage machine in the video game city actually caught the mobile phone blind box', 'If I had known you liked me, would we have']
info (string, 4569 distinct): ['network', 'Weibo', "weibo@YouTube [] official channel public number:Zhu Yi's Boring Life Bgm:Hu Weili", 'Tencent Video', 'weibo', "Weibo: Zhu Yi's Boring Life YouTube:[Zhu Yi's Boring Life] Official Channel Zhihu/Netease Cloud:Zhu Yi's boring life public number:Zhu Yi's boring life BGM:Hu Weili", 'Transfer from network', 'weibo@', "Wechat Official Account @weibo@ YouTube [Zhu Yi 's Boring Life Bgm:Hu Weili", '[Houda Official Cooperation Account]']
tags (string, 4672 distinct): ['mobile game, e-sports, on line, black material, anecdote, order, Baicao Garden, black technology, live to die, game, Ezreal, Mecha, poodle, Enemy, sequel, League of Legends, Sanwei Bookstore, Trivia', 'original singer, lightning strike, Work, Sun Honglei, cover, brainwashing cycle, philosophy, Mentor and friend, funny, ghost animal, masterpiece, manpower, happy heart, Vincent Fang, outside, Fill in the lyrics, kid', 'Miss Hua is handsome, hotspot, on line, arrogant, China, sea of stars, trump, cases, Sister Hua Niu, Information, American', 'golden ship, knight line, seiyuu, lightning strike, animation, Let the bullets fly, Movie, Jiang Wen, audience master, cardiopulmonary, ghost animal, scarlet letter, game, Xiang Yu, race girl, comprehensive', 'eyeball, animal synthesis, Thief, Aquarium, rabbit nest, aquarium, Humanity, Xiao Tian, American, life record', 'Root of all evil, crab boss, animation, Baicao Garden, Handwritten Dubbing for Short Films, dark circles, naruto, original animation, what the hell, anime spoof, blackhead', 'study, report error, cursor, plug-in, dog head, Design Ideas, alarm clock, mouse cursor, animation, Knowledge, creativity, stuck point, vision, blood pressure, Art, design, obsessive compulsive disorder, Mars, book bar, spirit cage, spoken language', 'Wangzai, Barrage, phone case, Lao Tzu, Life, playback volume, cell phone, high energy, daily, chocolate, self made', 'youth, the sisters, campus, clown, short film, love, emotion, sister in law, tornado, Boy friend, neck, Jay Chou, in love, girlfriend, atmosphere, Calcium Oxide, There is a love I want to talk to you, Life, Hua Xizi, daily, silly kids, self made', 'back of head, Jin Chen, Zhang Jike, good tits, team up, ping pong, physical education, black leather, Malone, sports, competitive sports, furry, original painting, pingpong, the sisters']
duration (numeric, 121 distinct): ['61', '92', '76', '124', '91', '151', '95', '79', '114', '119']
category (numeric, 8 distinct): ['3', '0', '4', '2', '5', '1', '6', '7']
I_think_most_people_are_basically_well_intentioned (numeric, 6 distinct): ['4', '3', '5', '2', '1', '0']
I_get_bored_with_crowded_parties (numeric, 6 distinct): ['3', '2', '4', '0', '1', '5']
I_m_a_person_who_takes_risks_and_breaks_the_rules (numeric, 6 distinct): ['3', '4', '2', '5', '1', '0']
I_like_adventure (numeric, 6 distinct): ['3', '4', '2', '5', '1', '0']
I_try_to_avoid_crowded_parties_and_noisy_environments (numeric, 6 distinct): ['3', '2', '4', '0', '1', '5']
I_like_to_plan_things_out_at_the_beginning (numeric, 6 distinct): ['3', '5', '4', '2', '1', '0']
I_worry_about_things_that_don_t_matter (numeric, 6 distinct): ['3', '2', '1', '0', '4', '5']
I_work_or_study_hard (numeric, 6 distinct): ['3', '4', '5', '2', '1', '0']
Although_there_are_some_liars_in_the_society__I_think_most_peopl (numeric, 6 distinct): ['3', '4', '5', '2', '1', '0']
I_have_a_spirit_of_adventure_that_no_one_else_has (numeric, 6 distinct): ['3', '2', '4', '5', '1', '0']
I_often_feel_uneasy (numeric, 6 distinct): ['3', '2', '0', '1', '4', '5']
I_m_always_worried_that_something_bad_is_going_to_happen (numeric, 6 distinct): ['3', '2', '0', '1', '4', '5']
Although_there_are_some_dark_things_in_human_society__such_as_wa (numeric, 6 distinct): ['3', '4', '5', '2', '1', '0']
I_enjoy_going_to_social_and_entertainment_gatherings (numeric, 6 distinct): ['3', '4', '2', '5', '1', '0']
It_is_one_of_my_characteristics_to_pay_attention_to_logic_and_or (numeric, 6 distinct): ['3', '4', '5', '2', '1', '0']
'''

CONTEXT = "REASONER Recommendation Dataset"
TARGET = CuratedTarget(raw_name="like", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []