from tabular.datasets.manual_curation_obj import CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: hernan4444/animeplanet-recommendation-database-2020
====
Examples: 16621
====
URL: https://www.kaggle.com/hernan4444/animeplanet-recommendation-database-2020
====
Description: 
Anime-Planet Recommendation Database 2020
Recommendation data from 74.000 users and 16.000 animes at Anime-Planet

This dataset contains information about 16.621 anime, 175.731 recommendations and the preference from 74.129 different users of animes scrapped from anime-planet. In particular, this dataset contain:

Information about the anime like Tags, synopsis, average score, etc.
List of animes recommended given another anime and the count of user that are agreed with the recommendation.
HTML with anime information to do data scrapping. These files contain information such as reviews, synopsis, information about the staff, anime statistics, genre, etc.
the anime list per user. Include dropped, watched, want to watch, currently watching, stalled and Won't watch.
ratings given by users to the animes that they has watched completely.

Warning: this dataset includes information about anime for adults (hentai).
Content
The anime data was scrapped between June 4th and June 25th.

The "html" folder contain 1 zip per anime (16.621 different anime). Each zip contains different HTML pages scrapped from Anime-planet. The scrapped pages are:
Main page
Reviews
Recommendations
Characters
Staff
I uploaded 2 files as example to don't increase the size of this dataset. All HTML files are in this link: https://drive.google.com/drive/folders/1xIxBRtJR2oTZhJVvjFoTo3qllBFn4aOV?usp=sharing

animelist.csv have the list of all animes register by the user with the respective score, watching status and numbers of episodes watched. This dataset contains 20 Million row, 16.745 different animes and 74.129 different users. The file have the following columns:
user_id: non identifiable randomly generated user id.
anime_id: Anime-planet ID of the anime. (e.g. 1).
score: score between 1 to 5 given by the user in scale of 0.5. 0 if the user didn't assign a score. (e.g. 3.5)
watching_status: state ID from this anime in the anime list of this user. (e.g. 2)
watched_episodes: numbers of episodes watched by the user. (e.g. 24)
watching_status.csv describe every possible status of the column: "watching_status" in animelist.csv.

rating_complete.csv is a subset of animelist.csv. This dataset only considers animes that the user has watched completely (watching_status==1) and gave it a score (score!=0). This dataset contains 8 Million ratings applied to 15.681 animes by 68.199 users. This file have the following columns:

user_id: non identifiable randomly generated user id.
anime_id: Anime-planet ID of the anime. (e.g. 1).
rating: rating that this user has assigned.
anime_recommendations.csv have the list of all animes recommended given one anime. This information was scrapped from "recommendation" tab (e.g. https://www.anime-planet.com/anime/the-saints-magic-power-is-omnipotent/recommendations ). The file have the following columns:
Anime: Anime Planet ID of the anime. (e.g. 1).
Recommendation: Anime Planet ID of the recommended anime. (e.g. 1).
Agree Votes: number of users that was agreed with the recommendation.

anime.csv contain general information of every anime (16.621 different anime) like Tags, type, studio, synopsis, etc. This file have the following columns:

Anime-PlanetID: Anime Planet ID of the anime. (e.g. 1).
Name: full name of the anime. (e.g. FLCL)
Alternative Name: another way to call the anime. (e.g. Furi Kuri)
Rating Score: average score of the anime given from all users in Anime Planet database. (e.g. 8.78)
Number Votes: number of users who give a score to the anime. (e.g. 1241)
Tags: comma separated list of tags for this anime. (e.g. Comedy, Mecha, Sci Fi, Outer Space, Original Work)
Content Warning: comma separated list of content warning tags. (e.g. Explicit Violence, Mature Themes, Nudity)
Type: TV, movie, OVA, etc. (e.g. TV).
Episodes: number of chapters. (e.g. 26)
Finished: True if the anime finished when I did the data scraping. False is the anime is on going in that moment.
Duration: duration of the anime in minutes (e.g 60)
StartYear: year when the anime start the transmission. (e.g. 2016)
EndYear: year when the anime finish the transmission. (e.g. 2017)
Season: season and year of release (e.g. Fall 2000)
Studios: comma separated list of studios (e.g. Sunrise)
Synopsis: synopsis of the anime

Url: url to the main page of anime in Anime Planet (e.g. https://www.anime-planet.com/anime/vandread)
Acknowledgements
Thanks to:

Anime Planet for providing anime data.
Inspiration
Improve Anime Recommendation Database 2020 with more data like tags, content warning, another synopsis, etc.

Experiment with different types of recommended. For instance, collaborative filtering or based on context like Tags, synopsis, etc.

Use this information to build a better anime recommended system.

Identifying which feature allows us to build the best anime recommended system.

Build a second dataset with anime list per user.

====
Features:

Anime-PlanetID (int64, 16621 distinct): ['10', '4824', '4792', '4793', '4794', '4796', '4797', '4798', '4799', '48']
Name (object, 16619 distinct): ['[email\xa0protected]', 'The Prince of Tennis', 'Arashi no Yoru ni: Himitsu no Tomodachi', 'The Doraemons: Mushimushi Pyonpyon Daisakusen!', 'Dorami-chan: Hello Kyouryuu Kids!!', 'Goddamn', 'Himitsu no Akko-chan: Umi da! Obake da!! Natsu Matsuri', 'Koneko no Rakugaki', 'Manga Mitokoumon', 'Anmitsu-hime']
Alternative Name (object, 7290 distinct): ['Unknown', 'The Legend and The Hero', 'Long Zhi Gu: Jingling Wangzuo', 'Tennis no Ouji-sama', 'Grandpa Danger OVA', 'Alt titles: Bishoujo Senshi Sailor Moon (2014), Bishoujo Senshi Sailor Moon Crystal', 'Aura: Maryuuinkouga Saigo no Tatakai', 'Dakara Boku wa, H ga Dekinai.: Mie Sugi! Mizugi Contest', 'Kamisama Hajimemashita', 'Toaru Majutsu no Index Movie: Endymion no Kiseki']
Rating Score (object, 3517 distinct): ['Unknown', '2.25', '2.686', '2.95', '3.016', '2.652', '3.405', '3.409', '3.027', '2.519']
Number Votes (object, 4026 distinct): ['Unknown', '12', '10', '11', '13', '14', '15', '18', '16', '17']
Tags (object, 10782 distinct): ['Unknown', 'Vocaloid', 'Shorts', 'Minna no Uta', 'Family Friendly, Minna no Uta', 'Abstract', 'Abstract, Shorts', 'Chinese Animation, Family Friendly', 'Comedy', 'Abstract, No Dialogue, Shorts']
Content Warning (object, 207 distinct): ['Unknown', 'Violence', 'Nudity', 'Explicit Violence', 'Explicit Sex', 'Nudity, Sexual Content', 'Sexual Content', 'Nudity, Violence', 'Explicit Violence, Nudity', 'Mature Themes, Suicide']
Type (object, 8 distinct): ['TV', 'Movie', 'OVA', 'Web', 'Music', 'DVD', 'Other', 'TV\n(104']
Episodes (object, 215 distinct): ['1', '12', '13', '2', '26', 'Unknown', '3', '4', '6', '52']
Finished (bool, 2 distinct): ['1', '0']
Duration (object, 151 distinct): ['Unknown', '4', '2', '5', '3', '1', '6', '24', '10', '15']
StartYear (object, 104 distinct): ['2017', '2018', '2016', '2014', '2015', '2019', '2013', '2012', '2020', '2011']
EndYear (object, 104 distinct): ['2017', '2016', '2018', '2015', '2014', '2019', '2013', '2012', '2020', '2011']
Season (object, 111 distinct): ['Unknown', 'Spring 2018', 'Fall 2016', 'Spring 2016', 'Winter 2021', 'Spring 2017', 'Fall 2018', 'Spring 2021', 'Fall 2020', 'Fall 2017']
Studios (object, 1045 distinct): ['Unknown', 'Toei Animation', 'Sunrise', 'J.C.Staff', 'TMS Entertainment', 'MADHOUSE', 'Studio DEEN', 'Production I.G', 'Pierrot', 'A-1 Pictures']
Synopsis (object, 9067 distinct): ['No synopsis yet - check back soon!', "In 19th century Belgium, in the Flanders countryside, lived a young boy with an artistic flair named Nello, and his faithful companion Patrash. Though poor in the physical sense, the two friends shared a rich life along with Alois, one of Nello's neighbors, and his grandfather, his last living relative. Though great sorrow and hardship looms closely in the future, one thing is for certain, the devotion and companionship of Nello and Patrash will never fade...", 'The films will follow The First Summer of Love phenomenon that occurred a decade before the first Eureka Seven series. The anime franchise thus far has hinted at, but never depicted in full, "the beginning of it all." The films will then have the same basic story as the first Eureka Seven series, but will have an original ending. It will have completely re-recorded lines, redone footage, and new scenes.', 'The eccentric Suzumiya Haruhi wants nothing more than to meet aliens, time travelers and espers… but she’ll have to settle for the everyday Kyon instead! Along with the mysterious Itsuki and the vacant Mikuru, the duo forms the SOS Brigade – a club whose mission is to discover the mysteries of the world. Armed with a razor sharp wit and a skill for manipulation, Haruhi will stop at nothing to have fun at all costs, even at the expense of Mikuru’s dignity!', 'The original story follows the human drama of the Yamazaki family in Tokyo in the Year Showa 39 (1964) — the year that the city hosted the Summer Olympics.', "Life can be tough when you're a teenager. Enter Tsukino Usagi, an average, if somewhat clumsy, junior high student whose voracious appetite for sweets and capacity for tears are offset by her enthusiasm for life. Her normal existence is suddenly turned upside down when a talking cat named Luna comes into her life. Suddenly, Usagi finds herself with the ability to transform into the superhero known as Sailor Moon. Fighting the occasional monster may be the least of her worries, though...", 'The Holy Grail War is a battle between seven magicians who each summon a mythical hero to fight for their cause. Shirou, a twice orphaned high school boy, had so little magical talent that his foster father did not bother teaching him about the war and its meaning. Thanks to that lack of foresight, Shirou finds himself in a bit of a pinch when he accidentally summons a hero of the strongest class, and is sucked into the fray. The Grail grants the winner any wish they have. But driven by an unyielding sense of justice and self-sacrifice, for what will Shirou fight?', 'Since the ancient times, the Kannagi priestesses have used their swords, or Okatana, to exorcise the creatures known as Aratama that brought chaos upon the world of man. These maidens were known as Tojis. They are a special task force within the police. They are allowed to have their Okatana on their person because they are government officials, but they mostly consist of middle school and high school girls who go to one of five training schools throughout the country. Though they mostly live normal school lives, if they are given a mission, they take their Okatana and unleash their powers, fighting to protect the people. This spring, the top Tojis from five schools across the country have been gathered for a customary tournament where they will use their abilities and fight for the top position. As the many Tojis trained and prepared for the upcoming tournament, there was one girl who was even more determined than the others on improving her swordsmanship. What lies before the end of her Okatana?', 'In the future, androids live side by side with humans – but not as their equals, as their slaves. Though they look identical, these androids must display a holographic ring over their heads so the difference is clear. One day, a boy named Rikuo finds abnormal activity patterns in the logs of his own android, and alongside his friend Masaki, he sets forth to find where the android has been. Much to their surprise, the duo discovers a secret café known as Eve no Jikan with a single rule: within its walls, there must be no discrimination between humans and robots. In this place, androids appear to be human and are even displaying signs of independence – a trait that should not be possible. Rikou finds his perceptions increasingly challenged as he struggles to come to terms with his own android, and the relationship between man and machines...', 'During their travels through the Unova region, Ash and his friends Iris and Cilan arrive in Eindoak Town, built around a castle called the Sword of the Vale. The three Trainers have come to compete in the town’s annual battle competition, and Ash manages to win with some unexpected help from the Mythical Pokémon Victini! It turns out Victini has a special bond with this place... Long ago, the castle watched over the Kingdom of the Vale, and the partnership between Victini and the king protected the people who lived there. But that kingdom has since vanished into memory, leaving behind powerful relics and ancient Pokémon. Damon, a descendant of the People of the Vale, is trying to restore the lost kingdom with the help of his Reuniclus. His quest has taken him to the far reaches of the barren desert, and he has convinced the Legendary Pokémon Reshiram to join him! Damon plans to trap Victini and harness its power, and as that plan gets under way, the entire town of Eindoak faces disaster! Will the power of Ash’s ideals convince the Legendary Pokémon Zekrom to help stop Damon? Can they rescue Victini? The greatest adventure in Pokémon history approaches!']
Url (object, 16621 distinct): ['https://www.anime-planet.com/anime/the-prince-of-tennis', 'https://www.anime-planet.com/anime/hokuto-no-ken-legend-of-heroes', 'https://www.anime-planet.com/anime/dorami-chan-hello-kyouryuu-kids', 'https://www.anime-planet.com/anime/goddamn', 'https://www.anime-planet.com/anime/himitsu-no-akko-chan-umi-da-obake-da-natsu-matsuri', 'https://www.anime-planet.com/anime/koneko-no-rakugaki', 'https://www.anime-planet.com/anime/manga-mitokoumon', 'https://www.anime-planet.com/anime/anmitsu-hime', 'https://www.anime-planet.com/anime/berserk-golden-age-arc-ii-the-battle-for-doldrey', 'https://www.anime-planet.com/anime/haibane-renmei']
'''

CONTEXT = "Anime-Planet Recommendation Database 2020"
TARGET = CuratedTarget(raw_name='Rating Score', task_type=SupervisedTask.REGRESSION, numeric_missing="Unknown")
COLS_TO_DROP = ["Anime-PlanetID", "Url"]
FEATURES = []

DESCRIPTION = '''
Anime-Planet Recommendation Database 2020
Recommendation data from 74.000 users and 16.000 animes at Anime-Planet

This dataset contains information about 16.621 anime, 175.731 recommendations and the preference from 74.129 different users of animes scrapped from anime-planet. In particular, this dataset contain:

Information about the anime like Tags, synopsis, average score, etc.
List of animes recommended given another anime and the count of user that are agreed with the recommendation.
HTML with anime information to do data scrapping. These files contain information such as reviews, synopsis, information about the staff, anime statistics, genre, etc.
the anime list per user. Include dropped, watched, want to watch, currently watching, stalled and Won't watch.
ratings given by users to the animes that they has watched completely.

Warning: this dataset includes information about anime for adults (hentai).
Content
The anime data was scrapped between June 4th and June 25th.

The "html" folder contain 1 zip per anime (16.621 different anime). Each zip contains different HTML pages scrapped from Anime-planet. The scrapped pages are:
Main page
Reviews
Recommendations
Characters
Staff
I uploaded 2 files as example to don't increase the size of this dataset. All HTML files are in this link: https://drive.google.com/drive/folders/1xIxBRtJR2oTZhJVvjFoTo3qllBFn4aOV?usp=sharing

animelist.csv have the list of all animes register by the user with the respective score, watching status and numbers of episodes watched. This dataset contains 20 Million row, 16.745 different animes and 74.129 different users. The file have the following columns:
user_id: non identifiable randomly generated user id.
anime_id: Anime-planet ID of the anime. (e.g. 1).
score: score between 1 to 5 given by the user in scale of 0.5. 0 if the user didn't assign a score. (e.g. 3.5)
watching_status: state ID from this anime in the anime list of this user. (e.g. 2)
watched_episodes: numbers of episodes watched by the user. (e.g. 24)
watching_status.csv describe every possible status of the column: "watching_status" in animelist.csv.

rating_complete.csv is a subset of animelist.csv. This dataset only considers animes that the user has watched completely (watching_status==1) and gave it a score (score!=0). This dataset contains 8 Million ratings applied to 15.681 animes by 68.199 users. This file have the following columns:

user_id: non identifiable randomly generated user id.
anime_id: Anime-planet ID of the anime. (e.g. 1).
rating: rating that this user has assigned.
anime_recommendations.csv have the list of all animes recommended given one anime. This information was scrapped from "recommendation" tab (e.g. https://www.anime-planet.com/anime/the-saints-magic-power-is-omnipotent/recommendations ). The file have the following columns:
Anime: Anime Planet ID of the anime. (e.g. 1).
Recommendation: Anime Planet ID of the recommended anime. (e.g. 1).
Agree Votes: number of users that was agreed with the recommendation.

anime.csv contain general information of every anime (16.621 different anime) like Tags, type, studio, synopsis, etc. This file have the following columns:

Anime-PlanetID: Anime Planet ID of the anime. (e.g. 1).
Name: full name of the anime. (e.g. FLCL)
Alternative Name: another way to call the anime. (e.g. Furi Kuri)
Rating Score: average score of the anime given from all users in Anime Planet database. (e.g. 8.78)
Number Votes: number of users who give a score to the anime. (e.g. 1241)
Tags: comma separated list of tags for this anime. (e.g. Comedy, Mecha, Sci Fi, Outer Space, Original Work)
Content Warning: comma separated list of content warning tags. (e.g. Explicit Violence, Mature Themes, Nudity)
Type: TV, movie, OVA, etc. (e.g. TV).
Episodes: number of chapters. (e.g. 26)
Finished: True if the anime finished when I did the data scraping. False is the anime is on going in that moment.
Duration: duration of the anime in minutes (e.g 60)
StartYear: year when the anime start the transmission. (e.g. 2016)
EndYear: year when the anime finish the transmission. (e.g. 2017)
Season: season and year of release (e.g. Fall 2000)
Studios: comma separated list of studios (e.g. Sunrise)
Synopsis: synopsis of the anime

Url: url to the main page of anime in Anime Planet (e.g. https://www.anime-planet.com/anime/vandread)
Acknowledgements
Thanks to:

Anime Planet for providing anime data.
Inspiration
Improve Anime Recommendation Database 2020 with more data like tags, content warning, another synopsis, etc.

Experiment with different types of recommended. For instance, collaborative filtering or based on context like Tags, synopsis, etc.

Use this information to build a better anime recommended system.

Identifying which feature allows us to build the best anime recommended system.

Build a second dataset with anime list per user.
'''