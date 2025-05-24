from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: covertype
====
URL: https://www.openml.org/search?type=data&id=1596
====
Description: **Author**: Jock A. Blackard, Dr. Denis J. Dean, Dr. Charles W. Anderson  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Covertype) - 1998  

This is the original version of the famous covertype dataset in ARFF format. 

**Covertype**  
Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System &#40;RIS&#41; data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types). 

This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices. 

Some background information for these four wilderness areas: Neota (area 2) probably has the highest mean elevational value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational value, while Cache la Poudre (area 4) would have the lowest mean elevational value. 

As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5). Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4). 

The Rawah and Comanche Peak areas would tend to be more typical of the overall dataset than either the Neota or Cache la Poudre, due to their assortment of tree species and range of predictive variable values (elevation, etc.) Cache la Poudre would probably be more unique than the others, due to its relatively low elevation range and species composition.

Attribute Information:  
Given is the attribute name, attribute type, the measurement unit and a brief description. The forest cover type is the classification problem. The order of this listing corresponds to the order of numerals along the rows of the database. 
>
Name / Data Type / Measurement / Description  
Elevation / quantitative /meters / Elevation in meters  
Aspect / quantitative / azimuth / Aspect in degrees azimuth  
Slope / quantitative / degrees / Slope in degrees  
Horizontal_Distance_To_Hydrology / quantitative / meters / Horz Dist to nearest surface water features  
Vertical_Distance_To_Hydrology / quantitative / meters / Vert Dist to nearest surface water features  
Horizontal_Distance_To_Roadways / quantitative / meters / Horz Dist to nearest roadway  
Hillshade_9am / quantitative / 0 to 255 index / Hillshade index at 9am, summer solstice  
Hillshade_Noon / quantitative / 0 to 255 index / Hillshade index at noon, summer solstice  
Hillshade_3pm / quantitative / 0 to 255 index / Hillshade index at 3pm, summer solstice  
Horizontal_Distance_To_Fire_Points / quantitative / meters / Horz Dist to nearest wildfire ignition points  
Wilderness_Area (4 binary columns) / qualitative / 0 (absence) or 1 (presence) / Wilderness area designation  
Soil_Type (40 binary columns) / qualitative / 0 (absence) or 1 (presence) / Soil Type designation  
Cover_Type (7 types) / integer / 1 to 7 / Forest Cover Type designation 


Relevant Papers:  
- Blackard, Jock A. and Denis J. Dean. 2000. "Comparative Accuracies of Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables." Computers and Electronics in Agriculture 24(3):131-151. 
- Blackard, Jock A. and Denis J. Dean. 1998. "Comparative Accuracies of Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables." Second Southern Forestry GIS Conference. University of Georgia. Athens, GA. Pages 189-199. 
- Blackard, Jock A. 1998. "Comparison of Neural Networks and Discriminant Analysis in Predicting Forest Cover Types." Ph.D. dissertation. Department of Forest Sciences. Colorado State University. Fort Collins, Colorado. 165 pages.
====
Target Variable: class (nominal, 7 distinct): ['2', '1', '3', '7', '6', '5', '4']
====
Features:

Elevation (numeric, 1978 distinct): ['2968.0', '2962.0', '2991.0', '2972.0', '2975.0', '2978.0', '2988.0', '2955.0', '2952.0', '2965.0']
Aspect (numeric, 361 distinct): ['45.0', '0.0', '90.0', '135.0', '63.0', '315.0', '72.0', '18.0', '27.0', '34.0']
Slope (numeric, 67 distinct): ['11', '10', '12', '13', '9', '14', '8', '15', '16', '7']
Horizontal_Distance_To_Hydrology (numeric, 551 distinct): ['30.0', '0.0', '150.0', '60.0', '67.0', '42.0', '108.0', '85.0', '90.0', '120.0']
Vertical_Distance_To_Hydrology (numeric, 700 distinct): ['0.0', '3.0', '10.0', '7.0', '6.0', '13.0', '4.0', '5.0', '16.0', '9.0']
Horizontal_Distance_To_Roadways (numeric, 5785 distinct): ['150.0', '618.0', '900.0', '390.0', '1020.0', '990.0', '960.0', '997.0', '750.0', '1140.0']
Hillshade_9am (numeric, 207 distinct): ['226', '228', '230', '224', '223', '222', '233', '227', '225', '221']
Hillshade_Noon (numeric, 185 distinct): ['228', '231', '233', '229', '230', '234', '227', '223', '226', '225']
Hillshade_3pm (numeric, 255 distinct): ['143', '145', '138', '146', '142', '136', '139', '135', '149', '132']
Horizontal_Distance_To_Fire_Points (numeric, 5827 distinct): ['618.0', '541.0', '607.0', '942.0', '997.0', '700.0', '900.0', '726.0', '752.0', '960.0']
Wilderness_Area1 (nominal, 2 distinct): ['0', '1']
Wilderness_Area2 (nominal, 2 distinct): ['0', '1']
Wilderness_Area3 (nominal, 2 distinct): ['0', '1']
Wilderness_Area4 (nominal, 2 distinct): ['0', '1']
Soil_Type1 (nominal, 2 distinct): ['0', '1']
Soil_Type2 (nominal, 2 distinct): ['0', '1']
Soil_Type3 (nominal, 2 distinct): ['0', '1']
Soil_Type4 (nominal, 2 distinct): ['0', '1']
Soil_Type5 (nominal, 2 distinct): ['0', '1']
Soil_Type6 (nominal, 2 distinct): ['0', '1']
Soil_Type7 (nominal, 2 distinct): ['0', '1']
Soil_Type8 (nominal, 2 distinct): ['0', '1']
Soil_Type9 (nominal, 2 distinct): ['0', '1']
Soil_Type10 (nominal, 2 distinct): ['0', '1']
Soil_Type11 (nominal, 2 distinct): ['0', '1']
Soil_Type12 (nominal, 2 distinct): ['0', '1']
Soil_Type13 (nominal, 2 distinct): ['0', '1']
Soil_Type14 (nominal, 2 distinct): ['0', '1']
Soil_Type15 (nominal, 2 distinct): ['0', '1']
Soil_Type16 (nominal, 2 distinct): ['0', '1']
Soil_Type17 (nominal, 2 distinct): ['0', '1']
Soil_Type18 (nominal, 2 distinct): ['0', '1']
Soil_Type19 (nominal, 2 distinct): ['0', '1']
Soil_Type20 (nominal, 2 distinct): ['0', '1']
Soil_Type21 (nominal, 2 distinct): ['0', '1']
Soil_Type22 (nominal, 2 distinct): ['0', '1']
Soil_Type23 (nominal, 2 distinct): ['0', '1']
Soil_Type24 (nominal, 2 distinct): ['0', '1']
Soil_Type25 (nominal, 2 distinct): ['0', '1']
Soil_Type26 (nominal, 2 distinct): ['0', '1']
Soil_Type27 (nominal, 2 distinct): ['0', '1']
Soil_Type28 (nominal, 2 distinct): ['0', '1']
Soil_Type29 (nominal, 2 distinct): ['0', '1']
Soil_Type30 (nominal, 2 distinct): ['0', '1']
Soil_Type31 (nominal, 2 distinct): ['0', '1']
Soil_Type32 (nominal, 2 distinct): ['0', '1']
Soil_Type33 (nominal, 2 distinct): ['0', '1']
Soil_Type34 (nominal, 2 distinct): ['0', '1']
Soil_Type35 (nominal, 2 distinct): ['0', '1']
Soil_Type36 (nominal, 2 distinct): ['0', '1']
Soil_Type37 (nominal, 2 distinct): ['0', '1']
Soil_Type38 (nominal, 2 distinct): ['0', '1']
Soil_Type39 (nominal, 2 distinct): ['0', '1']
Soil_Type40 (nominal, 2 distinct): ['0', '1']
'''

CONTEXT = "Forest Cover Type Prediction from Cartographic Variables"
TARGET = CuratedTarget(raw_name="class", new_name="Forest Cover Type", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"1": "Spruce/Fir", "2": "Lodgepole Pine", "3": "Ponderosa Pine",
                                      "4": "Cottonwood/Willow", "5": "Aspen", "6": "Douglas-fir", "7": "Krummholz"})
COLS_TO_DROP = []
FEATURES = []
