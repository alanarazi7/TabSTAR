from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: maharshipandya/-spotify-tracks-dataset/dataset.csv
====
Examples: 114000
====
URL: https://www.kaggle.com/maharshipandya/-spotify-tracks-dataset/dataset.csv
====
Description: 
🎹 Spotify Tracks Dataset
A dataset of Spotify songs with different genres and their audio features

About Dataset
Content
This is a dataset of Spotify tracks over a range of 125 different genres. Each track has some audio features associated with it. The data is in CSV format which is tabular and can be loaded quickly.

Usage
The dataset can be used for:

Building a Recommendation System based on some user input or preference
Classification purposes based on audio features and available genres
Any other application that you can think of. Feel free to discuss!

Column Description
track_id: The Spotify ID for the track
artists: The artists' names who performed the track. If there is more than one artist, they are separated by a ;
album_name: The album name in which the track appears
track_name: Name of the track
popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
duration_ms: The track length in milliseconds
explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
loudness: The overall loudness of a track in decibels (dB)
mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.

====
Features:

Unnamed: 0 (int64, 114000 distinct): ['0', '75997', '76008', '76007', '76006', '76005', '76004', '76003', '76002', '76001']
track_id (object, 89741 distinct): ['6S3JlDAGk3uu3NtZbPnuhS', '2Ey6v4Sekh3Z0RUSISRosD', '2kkvB3RNRzwjFdGhaUA0tz', '5ZsAhuQ24mWHiduaxJqnhW', '08kTa3SL9sV6Iy8KLKtGql', '7tbzfR8ZvZzJEzy6v0d6el', '0YLSjVxSb5FT1Bo8Tnxr8j', '4WJTKbNJQ41zXnb84jSWaj', '2aaClnypAakdAmLw74JXxB', '2vU6bm5hVF2idVknGzqyPL']
artists (object, 31437 distinct): ['The Beatles', 'George Jones', 'Stevie Wonder', 'Linkin Park', 'Ella Fitzgerald', 'Prateek Kuhad', 'Feid', 'Chuck Berry', 'Håkan Hellström', 'OneRepublic']
album_name (object, 46589 distinct): ['Alternative Christmas 2022', 'Feliz Cumpleaños con Perreo', 'Metal', 'Halloween con perreito', 'Halloween Party 2022', 'The Complete Hank Williams', 'Fiesta portatil', 'Frescura y Perreo', 'Esto me suena a Farra', 'Perreo en Halloween']
track_name (object, 73608 distinct): ['Run Rudolph Run', 'Halloween', 'Frosty The Snowman', 'Little Saint Nick - 1991 Remix', 'Last Last', 'Christmas Time', 'CÓMO SE SIENTE - Remix', 'Sleigh Ride', 'RUMBATÓN', 'X ÚLTIMA VEZ']
popularity (int64, 101 distinct): ['0', '22', '21', '44', '1', '23', '20', '43', '45', '41']
duration_ms (int64, 50697 distinct): ['162897', '180000', '192000', '240000', '118840', '172342', '227520', '131733', '243057', '175986']
explicit (bool, 2 distinct): ['0', '1']
danceability (float64, 1174 distinct): ['0.647', '0.609', '0.579', '0.685', '0.602', '0.524', '0.689', '0.598', '0.607', '0.626']
energy (float64, 2083 distinct): ['0.876', '0.937', '0.931', '0.886', '0.801', '0.948', '0.858', '0.961', '0.92', '0.981']
key (int64, 12 distinct): ['7', '0', '2', '9', '1', '5', '11', '4', '6', '10']
loudness (float64, 19480 distinct): ['-5.662', '-4.457', '-9.336', '-7.57', '-4.034', '-8.871', '-3.725', '-4.324', '-5.08', '-12.472']
mode (int64, 2 distinct): ['1', '0']
speechiness (float64, 1489 distinct): ['0.0323', '0.0324', '0.0322', '0.0328', '0.0295', '0.0321', '0.033', '0.0367', '0.0326', '0.0306']
acousticness (float64, 5061 distinct): ['0.995', '0.993', '0.994', '0.992', '0.991', '0.131', '0.881', '0.108', '0.107', '0.99']
instrumentalness (float64, 5346 distinct): ['0.0', '0.0', '0.895', '0.905', '0.934', '0.922', '0.911', '0.0001', '0.913', '0.9']
liveness (float64, 1722 distinct): ['0.108', '0.111', '0.109', '0.11', '0.105', '0.107', '0.103', '0.106', '0.112', '0.113']
valence (float64, 1790 distinct): ['0.961', '0.304', '0.717', '0.962', '0.324', '0.963', '0.55', '0.365', '0.949', '0.202']
tempo (float64, 45653 distinct): ['0.0', '151.925', '95.004', '87.925', '130.594', '92.988', '125.004', '76.783', '77.321', '90.04']
time_signature (int64, 5 distinct): ['4', '3', '5', '1', '0']
track_genre (object, 114 distinct): ['acoustic', 'punk-rock', 'progressive-house', 'power-pop', 'pop', 'pop-film', 'piano', 'party', 'pagode', 'opera']
'''

CONTEXT = "Spotify Tracks Popularity"
TARGET = CuratedTarget(raw_name="popularity", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Unnamed: 0", "track_id"]
FEATURES = []

DESCRIPTION = '''
🎹 Spotify Tracks Dataset
A dataset of Spotify songs with different genres and their audio features

About Dataset
Content
This is a dataset of Spotify tracks over a range of 125 different genres. Each track has some audio features associated with it. The data is in CSV format which is tabular and can be loaded quickly.

Usage
The dataset can be used for:

Building a Recommendation System based on some user input or preference
Classification purposes based on audio features and available genres
Any other application that you can think of. Feel free to discuss!

Column Description
track_id: The Spotify ID for the track
artists: The artists' names who performed the track. If there is more than one artist, they are separated by a ;
album_name: The album name in which the track appears
track_name: Name of the track
popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
duration_ms: The track length in milliseconds
explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
loudness: The overall loudness of a track in decibels (dB)
mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
'''