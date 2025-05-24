from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: segment
====
Examples: 2310
====
URL: https://www.openml.org/search?type=data&id=40984
====
Description: **Author**: University of Massachusetts Vision Group, Carla Brodley  
**Source**: [UCI](http://archive.ics.uci.edu/ml/datasets/image+segmentation) - 1990  
**Please cite**: [UCI](http://archive.ics.uci.edu/ml/citation_policy.html)  

**Image Segmentation Data Set**
The instances were drawn randomly from a database of 7 outdoor images. The images were hand-segmented to create a classification for every pixel. Each instance is a 3x3 region.
 
__Major changes w.r.t. version 2: ignored first two variables as they do not fit the classification task (they reflect the location of the sample in the original image). The 3rd is constant, so should also be ignored.__


### Attribute Information  

4.  short-line-density-5:  the results of a line extractoin algorithm that 
          counts how many lines of length 5 (any orientation) with
          low contrast, less than or equal to 5, go through the region.
5.  short-line-density-2:  same as short-line-density-5 but counts lines
          of high contrast, greater than 5.
6.  vedge-mean:  measure the contrast of horizontally
          adjacent pixels in the region.  There are 6, the mean and 
          standard deviation are given.  This attribute is used as
         a vertical edge detector.
7.  vegde-sd:  (see 6)
8.  hedge-mean:  measures the contrast of vertically adjacent
           pixels. Used for horizontal line detection. 
9.  hedge-sd: (see 8).
10. intensity-mean:  the average over the region of (R + G + B)/3
11. rawred-mean: the average over the region of the R value.
12. rawblue-mean: the average over the region of the B value.
13. rawgreen-mean: the average over the region of the G value.
14. exred-mean: measure the excess red:  (2R - (G + B))
15. exblue-mean: measure the excess blue:  (2B - (G + R))
16. exgreen-mean: measure the excess green:  (2G - (R + B))
17. value-mean:  3-d nonlinear transformation
          of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals
          of Interactive Computer Graphics)
18. saturatoin-mean:  (see 17)
19. hue-mean:  (see 17)
====
Target Variable: class (nominal, 7 distinct): ['brickface', 'sky', 'foliage', 'cement', 'window', 'path', 'grass']
====
Features:

short.line.density.5 (numeric, 4 distinct): ['0.0', '0.1111', '0.2222', '0.3333']
short.line.density.2 (numeric, 3 distinct): ['0.0', '0.1111', '0.2222']
vedge.mean (numeric, 234 distinct): ['0.5', '0.3889', '1.0556', '1.3889', '1.2778', '1.1111', '1.2222', '0.8889', '0.0', '1.0']
vegde.sd (numeric, 1082 distinct): ['0.0', '0.0296', '0.1519', '0.2407', '0.1222', '0.1074', '0.0185', '0.1721', '0.063', '1.0628']
hedge.mean (numeric, 262 distinct): ['1.1111', '1.0', '0.6667', '1.0556', '0.8333', '1.3333', '0.0', '1.1667', '1.5', '1.2222']
hedge.sd (numeric, 1180 distinct): ['0.0', '0.1519', '0.0296', '0.3897', '0.063', '0.1721', '0.1361', '0.4741', '0.2074', '0.3443']
intensity.mean (numeric, 1271 distinct): ['0.0', '0.7407', '1.1482', '0.037', '1.3704', '0.7037', '6.2593', '16.1852', '6.4074', '6.4444']
rawred.mean (numeric, 681 distinct): ['0.0', '0.1111', '1.0', '7.5556', '0.5556', '0.2222', '0.3333', '7.4444', '6.5556', '0.4444']
rawblue.mean (numeric, 781 distinct): ['0.0', '6.4444', '6.6667', '7.2222', '7.6667', '7.3333', '7.4444', '9.0', '5.6667', '6.7778']
rawgreen.mean (numeric, 691 distinct): ['0.0', '0.1111', '3.0', '0.3333', '3.5556', '3.3333', '3.6667', '0.6667', '15.8889', '0.2222']
exred.mean (numeric, 430 distinct): ['0.0', '-14.3333', '-10.4444', '-12.0', '-9.2222', '-9.8889', '-1.4444', '-11.8889', '-4.2222', '-10.8889']
exblue.mean (numeric, 636 distinct): ['0.0', '3.1111', '3.3333', '25.4444', '-4.8889', '19.6667', '-3.4444', '1.7778', '0.2222', '4.1111']
exgreen.mean (numeric, 377 distinct): ['0.0', '-7.6667', '-14.2222', '-14.7778', '-15.7778', '-7.2222', '-13.4444', '-15.0', '-13.6667', '-8.0']
value.mean (numeric, 785 distinct): ['0.0', '7.6667', '7.7778', '7.2222', '7.5556', '8.0', '6.8889', '8.4444', '7.4444', '28.7778']
saturation.mean (numeric, 1899 distinct): ['1.0', '0.0', '0.9778', '0.1111', '0.9841', '0.9861', '0.2222', '0.7778', '0.8889', '0.5642']
hue.mean (numeric, 1922 distinct): ['-2.0944', '0.0', '-2.0712', '-2.1331', '-2.1233', '-2.0487', '-2.0131', '-2.0062', '-2.0562', '-2.0751']
'''

CONTEXT = "Outdoors Image Segmentation"
TARGET = CuratedTarget(raw_name="class", new_name="Image Class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []