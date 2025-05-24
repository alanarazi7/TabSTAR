from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: volcanoes-e1
====
Examples: 1183
====
URL: https://www.openml.org/search?type=data&id=1542
====
Description: **Author**: Michael C. Burl 
**Source**: UCI  
**Please cite**:   

* Dataset Title:  
Volcanoes on Venus - JARtool experiment Data Set  
Experiment: E1         

* Source:

Michael C. Burl 
MS 126-347, JPL 
4800 Oak Grove Drive 
Pasadena, CA 91109 
(818) 393-5345 
Michael.C.Burl '@' jpl.nasa.gov 
http://www-aig.jpl.nasa.gov/mls/home/burl/


* Data Set Information:  

The data was collected by the Magellan spacecraft over an approximately four year period from 1990--1994. The objective of the mission was to obtain global mapping of the surface of Venus using synthetic aperture radar (SAR). A more detailed discussion of the mission and objectives is available at JPL's Magellan webpage. 

There are some spatial dependencies. For example, background patches from with in a single image are likely to be more similar than background patches taken across different images. 

In addition to the images, there are "ground truth" files that specify the locations of volcanoes within the images. The quotes around "ground truth" are intended as a reminder that there is no absolute ground truth for this data set. No one has been to Venus and the image quality does not permit 100%, unambiguous identification of the volcanoes, even by human experts. There are labels that provide some measure of subjective uncertainty (1 = definitely a volcano, 2 = probably, 3 = possibly, 4 = only a pit is visible). See reference [Smyth95] for more information on the labeling uncertainty problem. 

There are also files that specify the exact set of experiments using in the published evaluations of the JARtool system. 

* Attribute Information:

The images are 1024X1024 pixels. The pixel values are in the range [0,255]. The pixel value is related to the amount of energy backscattered to the radar from a given spatial location. Higher pixel values indicate greater backscatter. Lower pixel values indicate lesser backscatter. Both topography and surface roughness relative to the radar wavelength affect the amount of backscatter.


* Relevant Papers:

G.H. Pettengill, P.G. Ford, W.T.K. Johnson, R.K. Raney, L.A. Soderblom, "Magellan: Radar Performance and Data Products", Science, 252:260-265 (1991). 

R.S. Saunders, A.J. Spear, P.C. Allin, R.S. Austin, A.L. Berman, R.C. Chandlee, J. Clark, A.V. Decharon, E.M. Dejong, "Magellan Mission Summary", J. of Geophysical Research Planets, 97(E8):13067-13090, (1992). 

M.C. Burl, L. Asker, P. Smyth, U. Fayyad, P. Perona, L. Crumpler, and J. Aubele, "Learning to Recognize Volcanoes on Venus", Machine Learning, (March 1998). 

P. Smyth, M.C. Burl, U.M. Fayyad, and P. Perona, Chapter: "Knowledge Discovery in Large Image Databases: Dealing with Uncertainties in Ground Truth", In Advances in Knowledge Discovery and Data Mining, AAAI/MIT Press, Menlo Park, CA, (1995).
====
Target Variable: Class (nominal, 5 distinct): ['1', '4', '3', '5', '2']
====
Features:

V1 (numeric, 669 distinct): ['534.0', '238.0', '290.0', '738.0', '20.0', '809.0', '773.0', '770.0', '160.0', '321.0']
V2 (numeric, 682 distinct): ['286.0', '196.0', '322.0', '486.0', '16.0', '168.0', '303.0', '609.0', '439.0', '655.0']
V3 (numeric, 1178 distinct): ['0.4824', '0.352', '0.352', '0.3869', '0.6834', '0.5621', '0.6047', '0.6174', '0.5289', '0.6048']
'''

CONTEXT = "Venus Volcanoes from Magellan Spacecraft"
TARGET = CuratedTarget(raw_name="Class", new_name="Is Volcano", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'Definitely', '2': 'Probably', '3': 'Possibly', '4': 'Only a pit', '5': 'No'})
COLS_TO_DROP = []
FEATURES = []