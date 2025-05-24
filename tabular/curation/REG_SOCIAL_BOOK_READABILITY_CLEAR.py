from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: verracodeguacas/clear-corpus/CLEAR.csv
====
Examples: 4726
====
URL: https://www.kaggle.com/verracodeguacas/clear-corpus/CLEAR.csv
====
Description: 
About Dataset
📚 CommonLit Ease of Readability (CLEAR) Corpus 📚

Dive into this comprehensive dataset, curated by CommonLit in collaboration with Georgia State University. With approximately 5,000 reading passages spanning from the 3rd to 12th grade levels, this resource is a treasure trove for researchers, educators, and data enthusiasts alike.

🔍 Key Features:

Unique readability scores for each passage.
Text excerpts covering over 250 years of literature across various genres.
Meta-data including publishing year, genre, and more.
Readability indices such as Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level, and predictions from the Kaggle Readability Prize.
An invaluable resource for NLP, education, and literary analysis.
🔄 Continuously updated by CommonLit, ensuring fresh and accurate data.

Whether you're developing a new readability metric, analyzing literary trends, or training a machine learning model, the CLEAR Corpus is your go-to dataset! 🚀

====
Features:

ID (float64, 4724 distinct): ['400.0', '5460.0', '5477.0', '5476.0', '5475.0', '5474.0', '5473.0', '5472.0', '5471.0', '5470.0']
Last Changed (float64, 1 distinct): ['6.01']
Author (object, 2409 distinct): ['simple wiki', 'wikipedia', '?', 'USHistory.org', 'CommonLit Staff', 'wikijunior', 'R.M. Ballantyne', 'Andrew Lang', 'Thomas Tapper', 'Thornton W. Burgess']
Title (object, 4658 distinct): ['Invention and Discovery', '?', 'Current History', 'LITTLE MISCHIEF', 'Bacteria', 'THE FAIRY GODMOTHERS', 'Protestant_Reformation', 'Fungus', 'Monitress Merle', 'Metabolism']
Anthology (object, 290 distinct): ['CLD', 'Frontiers for Young Minds', 'African Storybook Level 3', 'African Storybook Level 4', 'Boys and Girls Bookshelf; a Practical Plan of Character Building, Volume I', 'The European War, Vol. 1 - No. 6', 'African Storybook Level 5', 'Junior Classics Vol. 4', 'The European War, Vol. 1 - No. 5', 'Good Cheer Stories Every Child Should Know']
URL (object, 3688 distinct): ['https://www.africanstorybook.org/', 'https://www.africanstorybook.org/#', 'http://www.gutenberg.org/files/8742/8742-h/8742-h.htm', 'http://www.gutenberg.org/files/19721/19721-h/19721-h.htm', 'http://www.gutenberg.org/cache/epub/6323/pg6323-images.html', 'http://www.gutenberg.org/cache/epub/3152/pg3152.html', 'http://www.gutenberg.org/files/15417/15417-h/15417-h.htm', 'http://www.gutenberg.org/files/38280/38280-h/38280-h.htm', 'http://www.gutenberg.org/cache/epub/6577/pg6577-images.html', 'http://www.gutenberg.org/cache/epub/6302/pg6302-images.html']
Source (object, 19 distinct): ['gutenberg', 'kids.frontiersin', 'commonlit', 'simple.wikipedia', 'wikipedia', 'africanstorybook', 'online-literature', 'digitallibrary', 'freekidsbooks', 'static.ehe.osu.edu']
Pub Year (float64, 168 distinct): ['2020.0', '2019.0', '2017.0', '1915.0', '1881.0', '2018.0', '1883.0', '1882.0', '1922.0', '1914.0']
Category (object, 2 distinct): ['Lit', 'Info']
Location (object, 4 distinct): ['mid', 'start', 'whole', 'end']
License (object, 19 distinct): ['PD', 'CC BY 4.0', 'CC BY-SA 3.0', 'CC BY-SA 3.0 and GFDL', 'CC BY-NC-SA 2.0', 'CC BY 3.0', 'CC BY-NC-SA 4.0', 'CC BY-NC-SA 3.0', 'CC BY-NC-ND 4.0', 'CC BY-NC 3.0']
MPAA
Max (object, 4 distinct): ['G', 'PG', 'PG-13', 'R']
MPAA 
#Max (float64, 4 distinct): ['1.0', '2.0', '3.0', '4.0']
MPAA
#Avg (float64, 6 distinct): ['1.0', '1.5', '2.0', '2.5', '3.0', '4.0']
Excerpt (object, 4723 distinct): ['Presently the stars begin to peep out, timidly at first, as if to see whether the elements here below had ceased their strife, and if the scene on earth be such as they, from bright spheres aloft, may shed their sweet influences upon. Sirius, or that blazing world Argus, may be the first watcher to send down a feeble ray; then follow another and another, all smiling meekly; but presently, in the short twilight of the latitude, the bright leaders of the starry host blaze forth in all their glory, and the sky is decked and spangled with superb brilliants.\nIn the twinkling of an eye, and faster than the admiring gazer can tell, the stars seem to leap out from their hiding-places. By invisible hands, and in quick succession, the constellations are hung out; first of all, and with dazzling glory, in the azure depths of space appears the great Southern Cross. That shining symbol lends a holy grandeur to the scene, making it still more impressive.', 'When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.\nThe floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.\nAt each end of the room, on the wall, hung a beautiful bear-skin rug.\nThese rugs were for prizes, one for the girls and one for the boys. And this was the game.\nThe girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.\nThis would have been an easy matter, but each traveller was obliged to wear snowshoes.', 'On a number of the hills sat solemn old owls, trying to look very wise. Most of these owls sat perfectly still as we drove by; but I saw two or three fly slowly away, as if half asleep. I wonder if these sober old birds teach the little prairie-dogs any of their wisdom.\nAll the prairies in this part of Kansas are covered with a short, thick grass, called "buffalo-grass," and the dogs live on its roots. These roots are little bulbs, and make nice rich food for the funny little fellows.\nA gentleman who has lived here for many years tells me that all their houses are connected underground by halls or passages, so that they can travel a mile or so without coming to the top of the ground.\nWherever you see a prairie-dog village, there you will find good water by digging a few feet. Sometimes boys capture these odd little dogs, and they become quite tame and make cunning pets.', "So he put on a smile (of course it was not a very beautiful one, for he was in a hurry, but it was the best he could do), and stared straight into the cow's eyes. She saw that smile, and it so touched her that she stopped short. Then she sauntered back a little way, but the thought of that aggravating fly, and that awful frog, was too much for her poor nerves, and turning around, she dashed madly on again.\nIn another minute, the poor old man—cane, little legs, smile and all—was up in the air.\nHe alighted in the top of a hickory-tree. One branch grazed his eye, two ran into his legs, while another held his smile stiff and straight.\nThus he stayed until an eagle caught sight of him, pounced right down, and flew off with him to her nest, which was on a huge rock that rose straight up into the cold air and made the summit of a mountain.", 'When I look at pictures of people of old times, I often think what a curious thing it is that the only apparent difference between them and the people of the present day is to be seen in their clothes.\nIf we could take a dozen or so of ancient Greeks and Romans; some gentlemen and ladies of the middle ages; a party of our great-grandfathers and mothers, and some nice people who are now living in the next street, and were to dress all the women in calico frocks and sun-bonnets, and all the men in linen coats and trousers and broad straw hats, with their hair cut short; and were then to jumble them all up together, and make them keep their tongues quiet, it would be very difficult, if not impossible, for a committee, unacquainted with any of the party, to pick out the ancients, the middle-agers, or the moderns.', 'Now the whole of Daffydowndilly\'s life had hitherto been passed with his dear mother, who had a much sweeter face than old Mr. Toil, and who had always been very indulgent to her little boy. No wonder, therefore, that poor Daffydowndilly found it a woeful change to be sent away from the good lady\'s side and put under the care of this ugly-visaged schoolmaster, who never gave him any apples or cakes, and seemed to think that little boys were created only to get lessons.\n"I can\'t bear it any longer," said Daffydowndilly to himself, when he had been at school about a week. "I\'ll run away and try to find my dear mother; and, at any rate, I shall never find anybody half so disagreeable as this old Mr. Toil!"\nSo the very next morning, off started poor Daffydowndilly, and began his rambles about the world, with only some bread and cheese for his breakfast, and very little pocket-money to pay his expenses. But he had gone only a short distance when he overtook a man of grave and sedate appearance, who was trudging at a moderate pace along the road.', 'He understands fishing much better than most boys, for he seldom misses his game. He takes his position on the railing, and fixes his eyes upon the finny tribes below, and when a fish that suits him comes within his range, he dives into the water and brings it up with his stout beak, and then beats it upon the railing to make it limp and tender before swallowing.\nIt is not so very surprising that he is such an expert fisher, for during the winter it is his only occupation; he has no family to look after now, and he is so very selfish and quarrelsome that he will not allow any of his brothers to fish near him. He considers the whole length of the wharf his fishing-post, and his brothers must not trespass upon his grounds; if they do, he chases them away with a rattling, clanging noise, enough to frighten any fisher not stronger than himself.', '"One day Bella went to the city, and brought home a fine new bonnet in a large bandbox. During the evening she showed it with great pride to the young ladies; and, unknown to her, Jocko enjoyed the sight of the ribbons and laces and flowers from behind the parlor sofa.\n"Like Bella herself, he was fond of finery; and the bonnet seemed to him a very fit garment for a monkey to wear. So the next morning, while Bella was busy in the kitchen, Jocko went to her closet, took out her bandbox, dressed himself in the bonnet, and stole down the back-stairs.\n"Bella, hearing a noise, looked around, and there he was, his head literally lost in a sea of red and yellow ribbons. With a shout of rage, she seized the broomstick, and hurried after the thief. But before she could reach him, Jocko had mounted two flights of stairs, leaped out on the porch, and climbed up to the roof of the house.', 'Max, with his chubby hand, turned to the first page, and found the Christmas-tree, with the baby and flag at the top. Then mamma had to read the story, and, after it was finished, the same little hand turned the leaf back; for the blue eyes wanted to see baby Arthur again.\nThen how both pairs of eyes looked at Teddy with his new sled and, while mamma read to them the pretty verses of Teddy\'s mamma, they were still as mice.\nAnd how their eyes sparkled when they saw the picture of the wheelbarrows and cart loaded with earth! For this was just the way they used to play in the warm pleasant weather. They thought the three little boys must have had lots of fun.\nThen they wanted to hear about "Georgie\'s Pet Mouse," and "Bess and the Kitten." They did not wonder that "Baby" felt cross at having his picture taken; for Max had to sit still so long, and so many times for his, that he knew how to pity the poor baby.', 'So, the next day, the sugar being out, she bought two dollars\' worthwhile Teddy was at school, and without even telling his mother, she searched the house for a hiding-place. She shook her head at the pantry and cellar, but she visited the garret, and the spare front chamber; she looked into the camphor-chest, she contemplated a barrel of potatoes, she moved about the things in her wardrobe, and at last, she hid the sugar! No danger of Teddy finding it this time! Aunt Ann could not repress a smile of triumph as she sat down to her knitting.\nUnconscious Teddy came home at noon, ate his dinner, and was off again. His mother and Aunt Ann went out making calls that afternoon, and as Aunt Ann closed the street door she thought to herself—"I can really take comfort going out, I feel so safe in my mind, now that sugar is hid."']
Google
WC (float64, 76 distinct): ['179.0', '197.0', '189.0', '161.0', '177.0', '185.0', '191.0', '175.0', '192.0', '182.0']
Joon
WC v1 (float64, 86 distinct): ['193.0', '177.0', '190.0', '184.0', '198.0', '188.0', '189.0', '196.0', '183.0', '185.0']
British WC (float64, 8 distinct): ['0.0', '1.0', '2.0', '3.0', '5.0', '4.0', '9.0', '6.0']
British Words (object, 250 distinct): ['grey', 'travelled', 'centre', 'colour', 'axe', 'travelling', 'traveller', 'behaviour', 'programme', 'centres']
Sentence
Count v1 (float64, 38 distinct): ['8.0', '7.0', '6.0', '9.0', '10.0', '5.0', '11.0', '12.0', '13.0', '4.0']
Sentence
Count v2 (float64, 39 distinct): ['8.0', '7.0', '9.0', '6.0', '10.0', '11.0', '5.0', '12.0', '13.0', '14.0']
Paragraphs (float64, 18 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '10.0', '9.0']
BT Easiness (float64, 4724 distinct): ['-0.3403', '-2.2739', '-3.4311', '-0.7553', '-0.8033', '-0.5192', '-2.1419', '-0.0816', '-0.3571', '-0.4371']
BT s.e. (float64, 4724 distinct): ['0.464', '0.5147', '0.6002', '0.4847', '0.4646', '0.4792', '0.5265', '0.5072', '0.4814', '0.4626']
Flesch-Reading-Ease (float64, 3282 distinct): ['73.2', '61.33', '55.88', '75.69', '64.6', '75.43', '61.11', '57.61', '71.72', '58.03']
Flesch-Kincaid-Grade-Level (float64, 1593 distinct): ['10.14', '7.5', '11.81', '9.26', '11.52', '8.08', '10.37', '10.96', '11.39', '12.4']
Automated Readability Index (float64, 1789 distinct): ['10.43', '12.91', '13.79', '11.77', '11.32', '12.47', '11.38', '9.81', '12.14', '13.07']
SMOG Readability (float64, 29 distinct): ['11.0', '10.0', '12.0', '9.0', '8.0', '7.0', '13.0', '14.0', '6.0', '5.0']
New Dale-Chall Readability Formula (float64, 816 distinct): ['7.39', '7.89', '7.26', '7.36', '7.04', '6.98', '7.81', '7.65', '6.14', '6.91']
CAREC (float64, 4453 distinct): ['0.1549', '0.1507', '0.2075', '0.1015', '0.1802', '0.1917', '0.1439', '0.1594', '0.1725', '0.0977']
CAREC_M (float64, 4427 distinct): ['0.1617', '0.1795', '0.2069', '0.1177', '0.1444', '0.1749', '0.2708', '0.0549', '0.1386', '0.2272']
CARES (float64, 4718 distinct): ['0.5531', '0.4909', '0.6088', '0.4415', '0.4755', '0.4558', '0.3795', '0.3384', '0.4588', '0.5494']
CML2RI (float64, 4717 distinct): ['26.9921', '27.2612', '4.4395', '22.015', '16.7248', '8.6414', '8.3293', '10.8955', '14.7414', '5.2221']
firstPlace_pred (float64, 4724 distinct): ['-0.3838', '-1.865', '-3.0353', '-0.9768', '-0.4857', '-0.4865', '-1.4892', '-0.1985', '-0.0997', '-0.3168']
secondPlace_pred (float64, 4724 distinct): ['-0.2836', '-1.9822', '-3.2324', '-0.8806', '-0.4981', '-0.498', '-1.4676', '-0.169', '-0.0221', '-0.2903']
thirdPlace_pred (float64, 4724 distinct): ['-0.3469', '-2.0131', '-3.1908', '-0.7949', '-0.4409', '-0.3331', '-1.4036', '-0.0026', '0.0046', '-0.3879']
fourthPlace_pred (float64, 4724 distinct): ['-0.2816', '-2.1248', '-3.3357', '-0.8906', '-0.6295', '-0.3758', '-1.522', '-0.0421', '0.1979', '-0.2837']
fifthPlace_pred (float64, 4724 distinct): ['-0.2478', '-1.8251', '-3.0065', '-0.9552', '-0.4826', '-0.4722', '-1.5673', '-0.0965', '-0.0335', '-0.458']
sixthPlace_pred (float64, 4724 distinct): ['-0.2899', '-1.9218', '-3.1518', '-0.7854', '-0.5329', '-0.4026', '-1.4784', '-0.1129', '0.0763', '-0.4108']
Kaggle split (object, 2 distinct): ['Train', 'Test']
'''

CONTEXT = "Readability scores for text passages spanning various genres and time periods"
TARGET = CuratedTarget(raw_name="New Dale-Chall Readability Formula", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['Kaggle split', 'firstPlace_pred', 'secondPlace_pred', 'thirdPlace_pred', 'fourthPlace_pred',
                'fifthPlace_pred', 'sixthPlace_pred', "ID", "Last Changed"]
FEATURES = []

DESCRIPTION = '''
About Dataset
📚 CommonLit Ease of Readability (CLEAR) Corpus 📚

Dive into this comprehensive dataset, curated by CommonLit in collaboration with Georgia State University. With approximately 5,000 reading passages spanning from the 3rd to 12th grade levels, this resource is a treasure trove for researchers, educators, and data enthusiasts alike.

🔍 Key Features:

Unique readability scores for each passage.
Text excerpts covering over 250 years of literature across various genres.
Meta-data including publishing year, genre, and more.
Readability indices such as Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level, and predictions from the Kaggle Readability Prize.
An invaluable resource for NLP, education, and literary analysis.
🔄 Continuously updated by CommonLit, ensuring fresh and accurate data.

Whether you're developing a new readability metric, analyzing literary trends, or training a machine learning model, the CLEAR Corpus is your go-to dataset! 🚀
'''