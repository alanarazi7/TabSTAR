from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: noorrizki/top-korean-drama-list-1500/kdrama_list.csv
====
Examples: 1647
====
URL: https://www.kaggle.com/noorrizki/top-korean-drama-list-1500/kdrama_list.csv
====
Description: 
Top Korean Drama List (~1500)
This dataset contains 1646 data top kdrama's from MyDramalist (Apr 2023)

About Dataset
This dataset contains information about 1,646 Korean dramas obtained from web scraping the website https://mydramalist.com/ in April 2023. The dataset consists of 1646 rows and 10 columns: Title, Year, Score, Synopsis, URL, Cast, Rating, Network, Genre, and Tags. The dataset can be used for various analysis and research purposes related to Korean dramas.

Acknowledgements
This data is taken from the website https://mydramalist.com/shows/top?page=1 , I would like to express our gratitude to MyDramaList.com for providing a comprehensive source of information for Kdrama enthusiasts. Their platform has been instrumental in helping me compile this dataset.

====
Features:

Unnamed: 0 (int64, 1647 distinct): ['0', '1094', '1104', '1103', '1102', '1101', '1100', '1099', '1098', '1097']
Name (object, 1642 distinct): ['Hero', 'Trap', 'Hyena', 'Crazy Love', 'Once Again', 'One More Time', 'Sometoon 2020', 'Love Returns', 'Conspiracy in the Court', 'Gourmet']
Year (int64, 26 distinct): ['2022', '2019', '2021', '2020', '2018', '2017', '2016', '2015', '2014', '2013']
Genre (object, 584 distinct): ['Comedy, Romance, Drama', 'Romance, Drama, Melodrama', 'Comedy, Romance', 'Romance, Drama', 'Comedy, Romance, Life, Drama', 'Comedy, Romance, Youth, Drama', 'Romance', 'Business, Comedy, Romance, Drama', 'Romance, Youth, Drama', 'Comedy, Romance, Drama, Fantasy']
Main Cast (object, 1639 distinct): ['Sung Hoon, Kwon Yu Ri, Shim Hye Jin, Joo Jin Mo, Tae Hang Ho, Ji Soo Won', 'Park Si Young, Choi Hyun Wook, Lee Won Jung, Heo Won Seo, Park Ji Young, Kim Do Ah', 'Yoo Ji Tae, Park Hae Soo, Jeon Jong Seo, Lee Won Jong, Kim Ji Hoon, Jang Yoon Ju', 'Kim Ji Soo, Jung Chae Yeon, Jung Jin Young, Choi Ri, Kang Tae Oh, Hong Ji Yoon', 'Song Hye Kyo, Lee Do Hyun, Im Ji Yeon, Yeom Hye Ran, Park Sung Hoon, Jung Sung Il', 'Sung Hoon, Lee Ga Ryeong, Lee Tae Gon, Park Joo Mi, Jeon Soo Kyung, Jeon Noh Min', 'Jo Jung Suk, Yoo Yeon Seok, Jung Kyung Ho, Kim Dae Myung, Jeon Mi Do, Shin Hyun Bin', 'Yim Si Wan, Son Hyun Joo, Go Ah Sung, Park Yong Woo, Jeon Ik Ryung, Moon Won Ju', 'Jeon Hee Jin, Kim Hyun Jin, Jo Ha Seul, Choi Dae Soo, Kim Jong Hun, ViVi', 'Eugene, Jo Hyun Jae, Jae Hee, Shin Sung Rok, Yoon Sang Hyun, Kim Bin Woo']
Sinopsis (object, 1640 distinct): ['Do you remember when we met for the first time? Hyun Jin - a girl who finally meets her coveted guy. Hee Jin - a girl that is still in love. Ha Seul - a girl who is a childhood friend. ,', 'Thieves overtake the mint of a unified Korea. With hostages trapped inside, the police must stop them — as well as the shadowy mastermind behind it all. Remake of the Spanish TV series "Money Heist" (La Casa de Papel) 2017. ,', 'Han Geu Roo is an autistic 20-year-old. He works for his father’s business “Move To Heaven,” a company that specializes in crime scene cleanup, where they also collect and arrange items left by deceased people, and deliver them to the bereaved family. When Geu Roo\\s father dies, Geu Roo\\s guardianship passes to his uncle, ex-convict Cho Sang Gu, who is a martial arts fighter in underground matches. Per the father\\s will, Sang Gu must care for and work with Geu Roo in “Move To Heaven” for three months to gain full guardianship and claim the inheritance. Eying money, Sang Gu agrees to the conditions and moves in. Adapted from the nonfiction essay "Things Left Behind" by professional trauma cleaner Kim Sae Byul., Gu must care for and work with Geu Roo in “Move To Heaven” for three months to gain full guardianship and claim the inheritance. Eying money, Sang Gu agrees to the conditions and moves in. Adapted from the nonfiction essay "Things Left Behind" by professional trauma cleaner Kim Sae Byul.', 'After spending two years teaching in the country, Lee Min Woo returns to a city hospital to complete his residency and face his own uncertainties about being a doctor. Free-spirited and goofy, he is jaded about his job and just wants it easy. He is then jolted out of apathy when a traumatic incident forces him to rethink why he wanted to be a doctor in the first place. The first-year resident Kang Jae In comes from a rich family that owns hospitals, but she just simply wants to be a doctor who can support herself and help others. Both Min Woo and Jae In work in ER under the guidance of Choi In Hyuk, a famously astute workaholic surgeon who puts his patients before himself., Kang Jae In comes from a rich family that owns hospitals, but she just simply wants to be a doctor who can support herself and help others. Both Min Woo and Jae In work in ER under the guidance of Choi In Hyuk, a famously astute workaholic surgeon who puts his patients before himself.', '"Shim Jae Bok has always believed she lived life to the fullest. But, misfortunes never come singly. Her beloved husband ends up cheating on her. Whats worse, her benefactor stabs her in the back. Things are closing in on her, but shes not the type of person who gives up. Lets find out how she comes up against the harsh reality of life! ",', 'A drama tells the story of 7 idol trainees and their daily hardships. ,', 'Yoo Tan is the leader and vocalist of an indie band called One More Time, a band he started with his childhood friends ten years ago. The indie band flourished for a while even boldly refusing to succumb to the establishment at one point. But Tan is getting older, the popularity of his band is dwindling, and life isn’t getting any easier financially, so he eventually signs with a music label. While enduring the difficult conditions inherent at a major music label, an unforeseeable event takes place sweeping Tan up in it: An unwanted time leap which allows him to journey back in time to regain his girlfriend., enduring the difficult conditions inherent at a major music label, an unforeseeable event takes place sweeping Tan up in it: An unwanted time leap which allows him to journey back in time to regain his girlfriend.', 'It is the love triangle romance that takes place when Ye Jin goes to search for her long-time online friend named "Sweet Brick". She does not know if he is a member of her craftwork class named No Woon or the man she often meets by chance named Cha Ian. Adapted partially from the webtoon "Sometoon x OH MY GIRL". ,', 'The drama follows the life a woman who ends up losing everything after living a turbulent life. When she starts anew from the bottom, ironically, her life blossoms. The value that holds us together is not blood nor law, but rather love and affection between us. ,', 'Lee Na Young was trained to be an assassin after being rescued from servitude when her high ranking father was convicted of treason. Her childhood sweetheart Park Sang Kyu who had returned from studying abroad and worked as a government official was looking all over for her. So did her former servant Yang Man Oh who had become the leader of the merchant group. The three got entangled in a conspiracy against the Emperor who was determined to carry out reforms to create a new Korea whereby the class system would be eliminated. (DW), the Emperor who was determined to carry out reforms to create a new Korea whereby the class system would be eliminated. (DW)']
Score (float64, 28 distinct): ['7.4', '7.5', '7.3', '7.9', '7.8', '7.6', '7.7', '7.2', '8.1', '7.1']
Content Rating (object, 6 distinct): ['15+ - Teens 15 or older', 'Not Yet Rated', '13+ - Teens 13 or older', '18+ Restricted (violence & profanity)', 'G - All Ages', 'R - Restricted Screening (nudity & violence)']
Tags (object, 1622 distinct): ['Soap Opera', 'Adapted From A Manhwa', 'Short Length Series,, Miniseries,, Web Series', 'Miniseries', 'Uncle-Nephew Relationship,, Autism,, Death,, Savant Syndrome,, Mourning,, Tearjerker,, Life Lesson,, Cleaning And Organizing,, Autism Spectrum Disorder,, Murder', 'Short Length Series,, Animal Lover,, Miniseries,, Nice Male Lead', 'Orphan Female Lead,, Weak Female Lead,, Enemies To Lovers,, Love Square,, Second Chance,, First Love,, Filmed Abroad,, Spring Setting,, "Childhood Friends Relationship,", Pianist Male Lead', 'Eunuch Supporting Character,, Eunuch Male Lead,, Queen Supporting Character,, Queen Female Lead,, Historical Fiction,, Strong Male Lead,, Multiple Mains,, Tearjerker,, Joseon Dynasty', 'Student Female Lead,, Student Male Lead,, School Setting,, "Childhood Friends Relationship,", Teenager Female Lead,, Teenager Male Lead,, Short Length Series,, Reverse-Harem,, Multiple Mains,, Miniseries', 'Dark Fiction,, Femme Fatale,, Historical Fiction,, Qing Dynasty,, Antihero,, Harem,, Joseon Dynasty']
Network (object, 373 distinct): ['Viki', 'Netflix', 'Apple TV, Viki', 'KBS World', 'SBS World, Viki', 'Viki, Netflix', 'WeTV', 'iQIYI, Viki', 'Viki, WeTV', 'iQIYI']
img url (object, 1647 distinct): ['https://i.mydramalist.com/Rle36_4c.jpg?v=1', 'https://i.mydramalist.com/JBw0wc.jpg?v=1', 'https://i.mydramalist.com/14yyQc.jpg?v=1', 'https://i.mydramalist.com/jWXOBc.jpg?v=1', 'https://i.mydramalist.com/4qvkK_4c.jpg?v=1', 'https://i.mydramalist.com/q6E3B_4c.jpg?v=1', 'https://i.mydramalist.com/55Dv0c.jpg?v=1', 'https://i.mydramalist.com/NdpWAc.jpg?v=1', 'https://i.mydramalist.com/akAzmc.jpg?v=1', 'https://i.mydramalist.com/2dVP2c.jpg?v=1']
Episode (object, 97 distinct): ['16 episodes', '20 episodes', '12 episodes', '8 episodes', '10 episodes', '32 episodes', '6 episodes', '24 episodes', '4 episodes', '18 episodes']
'''

def get_episodes(episode: str) -> int:
    episode = episode.lower()
    assert episode.endswith("episodes"), f"Invalid episode format: {episode}"
    episode = episode.replace("episodes", "").strip()
    return int(episode)

CONTEXT = "Korean Dramas"
TARGET = CuratedTarget(raw_name="Score", task_type=SupervisedTask.REGRESSION)
# TODO: this can be nice for image datasets in the future
COLS_TO_DROP = ["Unnamed: 0", "img url"]
FEATURES = [CuratedFeature(raw_name="Episode", new_name="Number of Episodes", processing_func=get_episodes,
                           feat_type=FeatureType.NUMERIC),]

DESCRIPTION = '''
Top Korean Drama List (~1500)
This dataset contains 1646 data top kdrama's from MyDramalist (Apr 2023)

About Dataset
This dataset contains information about 1,646 Korean dramas obtained from web scraping the website https://mydramalist.com/ in April 2023. The dataset consists of 1646 rows and 10 columns: Title, Year, Score, Synopsis, URL, Cast, Rating, Network, Genre, and Tags. The dataset can be used for various analysis and research purposes related to Korean dramas.

Acknowledgements
This data is taken from the website https://mydramalist.com/shows/top?page=1 , I would like to express our gratitude to MyDramaList.com for providing a comprehensive source of information for Kdrama enthusiasts. Their platform has been instrumental in helping me compile this dataset.
'''