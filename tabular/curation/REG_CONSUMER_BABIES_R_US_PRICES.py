from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: REG_CONSUMER_BABIES_R_US_PRICES
====
Examples: 5085
====
URL: http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv
====
Description: 
Information of baby products scraped from the Babies R Us website. The task is to predict the price of baby products.

====
Features:

int_id (int64, 5085 distinct): ['1', '3389', '3396', '3395', '3394', '3393', '3392', '3391', '3390', '3388']
ext_id (int64, 5085 distinct): ['54717766', '63496256', '68448696', '49755856', '68448546', '68448586', '46912416', '68448576', '60203856', '43506636']
title (object, 4998 distinct): ['Minene Muslin Squares 2-Pack', "Marmont Hill - 'Big Brown Bear 2' Eric Carle Framed Art Print", "Marmont Hill - 'Zebras 2' Eric Carle Framed Art Print", "Marmont Hill - 'Snoopy All-Star 1950' Peanuts Framed Art Print", 'Bacati Circles and Stripes Hamper', "Marmont Hill - 'Yellow Sunflower' Eric Carle Framed Art Print", "Marmont Hill - 'Belle and Snoopy' Peanuts Print on Canvas", 'Minene Muslin Squares 3-Pack', 'Bacati Quilted Circles Changing Pad Cover', "Marmont Hill - 'Woodstock Playing Ball' Peanuts Framed Art Print"]
SKU (object, 5080 distinct): ['B070ADAD', '5A1B906A', '7A81E93F', 'E557CC3A', 'FCA7E9B9', '2B941CD4', 'CD240FBC', '92152A40', '7844E43D', '4FF6FB87']
price (float64, 202 distinct): ['19.99', '29.99', '24.99', '14.99', '49.99', '17.99', '34.99', '39.99', '12.99', '59.99']
is_discounted (int64, 2 distinct): ['0', '1']
category (object, 10 distinct): ['Room Decor', 'Nursery Bedding / Blankets', 'Nursery Bedding', 'Storage & Organization', 'Room Decor / Wall Decor', "Kids' Bedding / Twin & Full Bedding", 'Nursery Bedding / Sheets & Pads', "Kids' Bedding / Toddler Bedding", 'Room Decor / Wall Decor / Hanging Letters', "Kids' Bedding"]
company_struct (object, 193 distinct): ['Trend Lab', 'Sweet JoJo Designs', 'Babies R Us', 'RoomMates', 'Cotton Tale', 'One Grace Place', 'Marmont Hill', 'Bacati', 'Triboro Quilt Co.', 'Lambs & Ivy']
company_free (object, 180 distinct): ['Trend Lab', 'Sweet Jojo Designs', 'JoJo Designs', 'Lolli Living', 'aden', 'Majestic Home Goods', 'RoomMates! Simply', 'Northwest', 'Sadie & Scout', 'Pem America']
brand (float64, 0 distinct): []
weight (object, 14 distinct): [' 1.5 lbs', ' 0.5 lbs', ' 2 lbs', ' 1 lb. 5 oz.', ' 1 lb. 4 oz.', ' 9.4 oz', ' 9 oz', ' 8.6 oz', ' 3 lbs', ' 4 lbs']
length (object, 96 distinct): ['52"', '40"', 'Trend', '39"', '32"', '44"', '2)', '13"', '28"', 'in.']
width (object, 92 distinct): ['28"', '30"', '16"', '13"', '52"', '17.25"', '27"', '44"', '12"', 'includes:']
height (object, 52 distinct): ['BIGGS', '9"', 'in', '12"', '27"', 'inches', '9.5"', 'the', '26"', '15".']
fabrics (object, 48 distinct): ['cotton', 'polyester', 'cotton / polyester', 'plush / polyester', 'plush', 'cotton / muslin', 'plush / cotton', 'satin', 'microfiber / polyester', 'microfiber']
colors (object, 146 distinct): ['pink', 'blue', 'green', 'gray', 'black', 'grey', 'purple', 'chocolate', 'red', 'green / pink']
materials (object, 14 distinct): ['fleece', 'wood', 'microfiber', 'plastic', 'metal', 'wood / pine', 'phthalate', 'metal / plastic', 'polyurethane', 'velcro']
'''

CONTEXT = "Prices Prediction for baby product from Babies R Us website"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["int_id", "ext_id", "SKU"]
FEATURES = []

DESCRIPTION = '''
Information of baby products scraped from the Babies R Us website. The task is to predict the price of baby products.
'''