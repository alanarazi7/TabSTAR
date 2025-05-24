from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: women_clothing_review
====
Examples: 18788
====
URL: https://www.openml.org/search?type=data&id=46659
====
Description: Predict the score of a customer review of clothing items (sold by an anonymous retailer)
    based on the review text, how much positive feedback the review has received (numeric), and
    additional features about the product like its department (categorical). The data were collected by:
    https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: Rating (numeric, 5 distinct): ['5', '4', '3', '2', '1']
====
Features:

Unnamed: 0 (numeric, 18788 distinct): ['18255', '2681', '21355', '12786', '2277', '17179', '7945', '13664', '17219', '11218']
Clothing ID (numeric, 1101 distinct): ['1078', '862', '1094', '1081', '872', '829', '1110', '868', '895', '936']
Age (numeric, 77 distinct): ['39', '35', '36', '34', '38', '37', '41', '33', '46', '42']
Title (string, 11490 distinct): ['Love it!', 'Beautiful', 'Love!', 'Love', 'Beautiful!', 'Beautiful dress', 'Love this dress!', 'Love it', 'Perfect', 'Gorgeous']
Review Text (string, 18117 distinct): ["Perfect fit and i've gotten so many compliments. i buy all my suits from here now!", "I bought this shirt at the store and after going home and trying it on, i promptly went online and ordered two more! i've gotten multiple compliments anytime i wear any of them. great for looking put together with no fuss. \r\npeople that have commented there's were destroyed in the wash didn't read the care label which says dry clean.", 'I purchased this and another eva franco dress during retailer\'s recent 20% off sale. i was looking for dresses that were work appropriate, but that would also transition well to happy hour or date night. they both seemed to be just what i was looking for. i ordered a 4 regular and a 6 regular, as i am usually in between sizes. the 4 was definitely too small. the 6 fit, technically, but was very ill fitting. not only is the dress itself short, but it is very short-waisted. i am only 5\'3", but it fe', "Lightweight, soft cotton top and shorts. i think it's meant to be a beach cover-up but i'm wearing it as a thin, light-weight summer outfit on these hot hot days. the top has a loose elastic around the bottom which i didn't realize when i ordered it, but i like it and it matches the look in the photos. and the shorts are very low-cut - don't expect them up around your waist. again, i like that. some might want to wear a cami underneath because it's a thin cotton but i'm fine as-is. i bought it i", 'Definitely shorter than expected, but a super light, airy, and comfy dress for summer nights. a must for summer dinner parties!!!', 'I love this off the shoulder top. it is so flattering  and the  colors are gorgeous. i normally wear a small or medium in retailer tops, i am wearing an extra small in this one, and it fits great. i am only 5 1" tall so the length hits me perfectly.', "I just received them (blue) and wore them today. i got so many compliments! they are comfy and very flattering. i would not call the fabric cheap at all. it is super soft and washes and wears nicely. i ordered the medium and they fit perfectly, even though i am on the large side of medium. i'm ordering the other color right now!", "Yes, this runs big but i sized down from med to small and it works. i like the gray/white pattern. received compliments upon first wearing and know i'll wear it often. i knocked off a star because of the rayon fabric which wrinkles so easily. this top always looks rumpled in the back from sitting. still i'm glad to have it in my fall wardrobe.", 'The sweater is very comfy and looked good the first time i tried it on, however the material is thin and slightly see through.\nalso, after one wear it completely stretched out and made the long sleeves awkward to wear without washing in between each wear. after the third time i wore it, i discovered a large hole in the armpit, most likely due to the excessive washing. i would not recommend this sweater.', 'I love this dress. it is so soft and comfortable, perfect for summer!! i wish it came in more colors because i would buy everyone!!']
Recommended IND (numeric, 2 distinct): ['1', '0']
Positive Feedback Count (numeric, 79 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Division Name (string, 4 distinct): ['General', 'General Petite', 'Initmates']
Department Name (string, 7 distinct): ['Tops', 'Dresses', 'Bottoms', 'Intimate', 'Jackets', 'Trend']
Class Name (string, 20 distinct): ['Dresses', 'Knits', 'Blouses', 'Sweaters', 'Pants', 'Jeans', 'Fine gauge', 'Skirts', 'Jackets', 'Lounge']
'''

CONTEXT = "Women Clothing E-Commerce Reviews"
TARGET = CuratedTarget(raw_name="Rating", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["Unnamed:_0"]
FEATURES = []
