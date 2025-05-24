from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Bank-Note-Authentication-UCI
====
Examples: 1372
====
URL: https://www.openml.org/search?type=data&id=43466
====
Description: Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.
====
Target Variable: class (numeric, 2 distinct): ['0', '1']
====
Features:

variance (numeric, 1338 distinct): ['0.5706', '0.5195', '0.3292', '0.3798', '-2.6479', '-0.2062', '-1.8584', '0.9297', '-1.3', '-0.278']
skewness (numeric, 1256 distinct): ['-4.4552', '-3.2633', '0.7098', '-3.7971', '-0.0248', '10.1374', '9.5663', '9.2931', '9.2207', '0.0393']
curtosis (numeric, 1270 distinct): ['1.2421', '4.5718', '3.0895', '0.7572', '-3.7044', '-3.7867', '-1.6643', '2.8093', '4.6429', '-1.331']
entropy (numeric, 1156 distinct): ['-0.2957', '-0.9888', '-0.4444', '0.3612', '-0.2375', '-0.5621', '-7.5034', '0.3211', '-0.9849', '0.3948']
'''

CONTEXT = "Bank Note Authentication from images"
TARGET = CuratedTarget(raw_name="class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []