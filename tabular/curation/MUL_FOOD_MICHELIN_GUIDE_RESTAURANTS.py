from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ngshiheng/michelin-guide-restaurants-2021/michelin_my_maps.csv
====
Examples: 17735
====
URL: https://www.kaggle.com/ngshiheng/michelin-guide-restaurants-2021/michelin_my_maps.csv
====
Description: 
Michelin Guide Restaurants
A curated list of awesome restaurants from the Michelin Guide

Context
At the beginning of the automobile era, Michelin, a tire company, created a travel guide, including a restaurant guide.

Through the years, Michelin stars have become very prestigious due to their high standards and very strict anonymous testers. Michelin Stars are incredibly coveted. Gaining just one can change a chef's life; losing one, however, can change it as well.

The dataset is curated using Go Colly.

Content
The dataset contains a list of restaurants along with additional details (e.g. address, price range, cuisine type, longitude, latitude, etc.) curated from the MICHELIN Restaurants guide. The culinary distinctions (i.e. the 'Award' column) of the restaurants included are:

3 Stars
2 Stars
1 Star
Bib Gourmand
Selected Restaurants

Source Code
https://github.com/ngshiheng/michelin-my-maps

====
Features:

Name (object, 17068 distinct): ['Racines', 'Krone', 'Fleur de Sel', 'Terra', "L'Essentiel", 'Adler', 'Apicius', 'Il Ristorante - Niko Romito', 'Riva', 'La Torre']
Address (object, 17324 distinct): ['Emirates Palace Mandarin Oriental Hotel, West Corniche Road, Al Ras Al Akhdar, Abu Dhabi', 'Bernhard-Simon-Strasse 2, Bad Ragaz, 7310, Switzerland', '1100 15th St. NW, Washington, 20005, USA', '412 Greenwich St., New York, 10013, USA', 'Hermine-Bareiss-Weg 1, Baiersbronn, 72270, Germany', '2F, Good Nature Station, 318-6 Inaricho, Shimogyo-ku, Kyoto, 600-8022, Japan', 'Tonbachstraße 237, Baiersbronn, 72270, Germany', 'Höheweg 41, Interlaken, 3800, Switzerland', '8 avenue Dutuit, Paris, 75008, France', 'GF, North Esplanade, Wynn Palace, Avenida da Nave Desportiva, Cotai, Macau']
Location (object, 5795 distinct): ['Tokyo, Japan', 'Paris, France', 'London, United Kingdom', 'Singapore', 'New York, USA', 'Kyoto, Japan', 'Osaka, Japan', 'Seoul, South Korea', 'Bangkok, Thailand', 'Taipei, Taiwan']
Price (object, 36 distinct): ['€€', '€€€', '€€€€', '$$', '$$$', '$$$$', '¥¥¥', '$', '€', '¥¥']
Cuisine (object, 1664 distinct): ['Modern Cuisine', 'Traditional Cuisine', 'Creative', 'Japanese', 'Contemporary', 'Italian', 'French', 'Seafood', 'Modern British', 'Street Food']
Longitude (float64, 17671 distinct): ['103.8702', '103.8028', '103.8141', '113.2376', '135.4979', '127.0543', '103.8636', '103.8425', '12.3968', '101.6644']
Latitude (float64, 17673 distinct): ['1.3214', '37.5224', '1.2933', '1.324', '34.698', '23.1083', '1.3058', '39.9188', '1.3135', '39.9109']
PhoneNumber (float64, 17012 distinct): ['815031385225.0', '497442470.0', '97142555142.0', '971800323232.0', '85381188822.0', '41338282602.0', '33495716924.0', '34976239516.0', '6626599000.0', '842439878888.0']
Url (object, 17735 distinct): ['https://guide.michelin.com/en/monaco-region/monaco/restaurant/le-louis-xv-alain-ducasse-a-l-hotel-de-paris', 'https://guide.michelin.com/en/aragon/zaragoza/restaurant/bistronomo', 'https://guide.michelin.com/en/andorra-la-vella-region/andorra-la-vella/restaurant/celler-d-en-toni', 'https://guide.michelin.com/en/pais-vasco/vitoria-gasteiz/restaurant/andere', 'https://guide.michelin.com/en/comunidad-valenciana/xbia/restaurant/tosca462312', 'https://guide.michelin.com/en/principado-de-asturias/lastres/restaurant/eutimio', 'https://guide.michelin.com/en/canarias/puerto-de-la-cruz/restaurant/brunelli-s', 'https://guide.michelin.com/en/canarias/gua-de-isora/restaurant/txoko', 'https://guide.michelin.com/en/comunidad-de-madrid/madrid/restaurant/tori-key', 'https://guide.michelin.com/en/catalunya/girona/restaurant/sinofos']
WebsiteUrl (object, 15050 distinct): ['http://www.xinrongji.cc', 'http://www.bingsheng.com', 'https://www.bernard-loiseau.com/', 'https://www.brasseriecolette.de/', 'https://www.bei-schumann.de/', 'http://www.xinrongji.com', 'https://seven.ch/', 'https://danieletdenise.fr/', 'https://www.robuchon.jp/', 'https://saltandsilver.de/']
Award (object, 5 distinct): ['Selected Restaurants', 'Bib Gourmand', '1 Star', '2 Stars', '3 Stars']
GreenStar (int64, 2 distinct): ['0', '1']
FacilitiesAndServices (object, 885 distinct): ['Air conditioning', 'Air conditioning,Terrace', 'Air conditioning,Counter dining', 'Terrace', 'Air conditioning,Wheelchair access', 'Air conditioning,Terrace,Wheelchair access', 'Car park,Terrace', 'Cash only', 'Air conditioning,Car park', 'Air conditioning,Interesting wine list']
Description (object, 17733 distinct): ['Here, the second generation of the family sells local-style salad with pineapple, cucumber and deep-fried dough stick, tossed in a special home-made sauce and sprinkled with ground peanuts. Chilli is optional.', 'The chef mixes his own sensibilities into cuisine rooted in French bistro fare. The blackboard menu is a medley of authentic dishes and à la carte items made with seasonal ingredients. Generally, an appetiser and a main course is the perfect amount, but some items can be ordered as half portions, so ask your server.', 'With over 35 years of history behind it, this restaurant occupying a typical Galician house is a renowned local eatery. It features an attractive bar and wine cellar, with the floor upstairs set aside for the welcoming rustic-cum-contemporary dining room, where guests can choose from a traditional, seasonally inspired à la carte. Popular dishes on the menu include wild mushrooms and game, such as partridge, venison and wild boar. The owner is passionate about wine and it is not unusual for her to invite guests to her wine cellar so that they can select their bottle personally. An extensive set menu and two wine pairing options are also available.', 'Eutimio occupies an attractive property with a homely feel in a unique setting (although parking can occasionally be a problem). The traditionally focused à la carte is centred around fish and seafood, and is enhanced by interesting market-inspired daily suggestions. Several dishes require pre-ordering, such as the “caldereta” of fish and seafood, and the rice with lobster. The same goes for its hugely popular version of the traditional “cachopo” stew (the restaurant’s “Cacholetus” dish won Spain’s inaugural “cachopo” competition).', "A splendid steakhouse located close to the entrance to the Loro Parque zoo. This family-run restaurant is a must on the island for meat-lovers given its choice of top-quality meat, with the advantage of many being aged on the premises. Make sure you try Brunelli's steak tartare, its signature speciality, or one of its superb cuts of grilled meat (Uruguayan beef entrecôte, Nebraskas Black Angus T-Bone, German Simmental ribeye, Chateaubriand and Spanish Tomahawk), many of which are sold by weight and designed for two people to share. The particularly impressive picture window here overlooks the sea.", 'This restaurant has the feel of a modern Basque “txoko” (gastronomic society) and, in a nod to chef Martín Berasategui’s origins, offers cooking that is very “casual” in style. The glass-fronted terrace can be completely opened in the summer months.', 'A simple bistro that defines its cooking as haute-cuisine from the “barrio”! Contemporary-style tapas and raciones featuring Asian fusions, vegetables, tripe etc, alongside a more seasonally inspired menu.', 'A Japanese restaurant with a difference where the standard cold sushi is abandoned in favour of Yakitori-style cuisine, with a particular focus on grilled chicken kebabs.', 'This small, centrally located restaurant, that likes to view itself as a passageway (Tsuro in Japanese) between Jerez and Japan, is dominated by a bar at which the chef (the only employee) can be seen hard at work. This intense culinary experience reveals many of the secrets of Japanese cuisine, from the rice ceremony and its preparation to the importance of knife sharpening. The dining experience, which lasts around three hours, is based around an Omakase menu and changes depending on market availability. Booking is essential.', 'Located in the heart of the city, Andere is considered one of its classic addresses. Nowadays it sports a more modern look including an attractive, covered, winter garden - style patio. Traditional cuisine with a contemporary flourish.']
'''

CONTEXT = "Michelin Guide Restaurants Awards"
TARGET = CuratedTarget(raw_name="Award", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["Url", "WebsiteUrl"]
FEATURES = []

DESCRIPTION = '''
Michelin Guide Restaurants
A curated list of awesome restaurants from the Michelin Guide

Context
At the beginning of the automobile era, Michelin, a tire company, created a travel guide, including a restaurant guide.

Through the years, Michelin stars have become very prestigious due to their high standards and very strict anonymous testers. Michelin Stars are incredibly coveted. Gaining just one can change a chef's life; losing one, however, can change it as well.

The dataset is curated using Go Colly.

Content
The dataset contains a list of restaurants along with additional details (e.g. address, price range, cuisine type, longitude, latitude, etc.) curated from the MICHELIN Restaurants guide. The culinary distinctions (i.e. the 'Award' column) of the restaurants included are:

3 Stars
2 Stars
1 Star
Bib Gourmand
Selected Restaurants

Source Code
https://github.com/ngshiheng/michelin-my-maps
'''