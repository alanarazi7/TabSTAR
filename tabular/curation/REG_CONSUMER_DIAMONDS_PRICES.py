from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: diamonds
====
Examples: 53940
====
URL: https://www.openml.org/search?type=data&id=42225
====
Description: This classic dataset contains the prices and other attributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization.

Content
price price in US dollars (\$326--\$18,823)

carat weight of the diamond (0.2--5.01)

cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

color diamond colour, from J (worst) to D (best)

clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

x length in mm (0--10.74)

y width in mm (0--58.9)

z depth in mm (0--31.8)

depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

table width of top of diamond relative to widest point (43--95)
====
Target Variable: price (numeric, 11602 distinct): ['605', '802', '625', '828', '776', '698', '789', '544', '666', '552']
====
Features:

carat (numeric, 273 distinct): ['0.3', '0.31', '1.01', '0.7', '0.32', '1.0', '0.9', '0.41', '0.4', '0.71']
cut (nominal, 5 distinct): ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
color (nominal, 7 distinct): ['G', 'E', 'F', 'H', 'D', 'I', 'J']
clarity (nominal, 8 distinct): ['SI1', 'VS2', 'SI2', 'VS1', 'VVS2', 'VVS1', 'IF', 'I1']
depth (numeric, 184 distinct): ['62.0', '61.9', '61.8', '62.2', '62.1', '61.6', '62.3', '61.7', '62.4', '61.5']
table (numeric, 127 distinct): ['56.0', '57.0', '58.0', '59.0', '55.0', '60.0', '54.0', '61.0', '62.0', '63.0']
x (numeric, 554 distinct): ['4.37', '4.34', '4.33', '4.38', '4.32', '4.35', '4.39', '4.31', '4.36', '4.4']
y (numeric, 552 distinct): ['4.34', '4.37', '4.35', '4.33', '4.32', '4.39', '4.38', '4.4', '4.31', '4.41']
z (numeric, 375 distinct): ['2.7', '2.69', '2.71', '2.68', '2.72', '2.67', '2.73', '2.66', '2.74', '4.02']
'''

CONTEXT = "Diamond Prices"
TARGET = CuratedTarget(raw_name="price", new_name="Price in USD", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
