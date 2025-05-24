from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: jc_penney_products
====
Examples: 10860
====
URL: https://www.openml.org/search?type=data&id=46661
====
Description: Predict the sale price of items sold on the website of the retailer JC Penney based on text features
    like its title/description, and numeric features like its rating. Representing an important (e)commerce
    task, this data was originally collected using information from the online page for each product:
    https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: sale_price (numeric, 2402 distinct): ['36.25', '30.2', '24.16', '60.42', '18.12', '42.29', '48.33', '21.74', '33.83', '35.46']
====
Features:

name_title (string, 8610 distinct): ['Champion® Vapor Shorts - Big & Tall', 'Clarks® Leisa Grove Leather Sandals', 'Gold Toe® Mens Dress Crew Socks', 'Stafford® Gunner Mens Cap Toe Leather Boots', 'My Little Pony Short-Sleeve Tee - Toddler Girls 2t-4t', 'John Deere Womens Utility Work Boots', "Levi's® 501® Original Fit Jeans-Big & Tall", 'Stylus™ Flare Jeans', 'Champion® Vapor Short-Sleeve Tee - Big & Tall', 'Xersion™ Quick-Dri Performance Bootcut Pant']
description (string, 8440 distinct): ['Stay comfortable and looking great all day with these sandals that are crafted in luxurious leather. technology Cushion Soft technology offers softness you can feel from your first step, long-lasting comfort and fit with minimal cushion compression optimal breathability via open-cell technology OrthoLite® cushioned footbed absorbs impact and offers breathability, moisture management and is antimicrobial construction leather upper EVA sole details strappy open-toe design adjustable hook-and-loop closure fabric lining', 'Work out in our shorts, featuring an elastic waist and Champion Vapor moisture-wicking fabric to keep you cool and dry.\xa0 elastic waist with drawstring 2 pockets flat front Champion Vapor moisture-wicking fabric 11" inseam polyester washable imported', 'You’ve picked the ones that began the brand: the iconic Levi’s® 501® Original Fit Jean. Authentic, not only because of their heritage but because they fit right, look good, and tell a great story. Sitting higher at your waist to fit the 5-button fly, the regular cut straight leg is not too tight, or too loose, and has no added stretch - just durable heavyweight cotton. Worn by guys who define their own style; coveted by denim collectors the world over. \xa0 matching big & tall web id: 5834008 sits at waist classic seat and thighs straight leg, 16" opening button fly preshrunk to retain shape wash after wash cotton washable imported', 'Add polished panache to your professional looks with our hand-finished leather boots by Stafford. \xa0 full-grain leather upper 6" shaft full-length leather padded sock lining cap toe lace up flexible two-tone rubber sole \xa0', 'Add a little retro to your wardrobe to create fashion fun with our Stylus flare jeans. high rise flare leg 5-pockets 32" inseam cotton/polyester/rayon/spandex washable imported \xa0', 'Our tee features soft-hand fabric made with Champion Vapor moisture-wicking technology to keep you cool and dry.\xa0 crewneck short sleeves Champion Vapor moisture-wicking technology polyester washable imported', 'Whether you\'re working out or running—or heading to the coffee shop afterwards—our regular rise, slim-fit, bootcut pants will keep you comfortable. tight compression fit provides added comfort during workouts Quick-Dri® moisture-wicking fabric helps keep you dry no-chafe seams won\'t irritate your skin key pocket at inside waistband jersey knit polyester/spandex washable imported misses: 32" inseam petite: 29½" inseam', 'All the quality and details of the original 501® jean, made from rigid denim that you finish yourself. Keep it dark and clean by washing as little as possible or repeat washings to fade them to your liking.\xa0 \xa0 matching big & tall web id: 5834950 sits at waist button fly straight leg,\xa017" opening extra room in the seat classic 5-pocket styling cotton washable imported Special Ordering InstructionsDenim will shrink approximately 10%.Add 1" to waist size for waists 27"-36".Add 2" to waist size for waists 38"-40".', 'With a straight-leg fit that\'s not too loose and not too tight, it\'s easy to see why 514™ jeans are a Levi\'s® \xa0bestseller. \xa0 \xa0 sits below waist 2 concealed back hem pockets\xa0 wide side seam slim fit through seat and thighs straight leg, 16½" opening black, rigid grey, blue stone and black stone: 99% cotton/1% elastane midnight and white bull denim: 98% cotton/2% elastane covered up: 78% cotton/16% polyester/4% elamul/2% elastane other colors: 100% cotton washable imported', "This climacool crewneck tee is the ideal blend of performance and comfort, with a streamlined fit to complement your body's movement. climacool paneled zones for ventilation non-chafing stitching banded collar tag free polyester washable imported"]
average_product_rating (numeric, 35 distinct): ['5.0', '4.5', '4.7', '4.0', '4.8', '4.6', '4.4', '4.3', '4.9', '3.0']
brand (string, 949 distinct): ['ARIZONA', 'Asstd National Brand', 'LIZ CLAIBORNE', 'FINE JEWELRY', 'Nike', 'Xersion', 'STAFFORD', 'WORTHINGTON', "St. John's Bay", 'Levi']
total_number_reviews (numeric, 308 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
'''

CONTEXT = "JC Penney Product Prices in Retailer Website"
TARGET = CuratedTarget(raw_name="sale_price", new_name="Sale Price in USD", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
