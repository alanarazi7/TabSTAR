from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: dresses-sales
====
Examples: 500
====
URL: https://www.openml.org/search?type=data&id=23381
====
Description: **Author**: Muhammad Usman & Adeel Ahmed   
**Source**: origin source at [UCI](https://archive.ics.uci.edu/ml/datasets/Dresses_Attribute_Sales)  
**Please cite**: [Paper that claims to first have used the data](https://www.researchgate.net/profile/Dalia_Jasim/publication/293464737_main_steps_for_doing_data_mining_project_using_weka/links/56b8782008ae44bb330d2583/main-steps-for-doing-data-mining-project-using-weka.pdf)

####1. Summary

This dataset contain attributes of dresses and their recommendations according to their sales. Sales are monitor on the basis of alternate days. 

The attributes present analyzed are: Recommendation, Style, Price, Rating, Size, Season, NeckLine, SleeveLength, waiseline, Material, FabricType, Decoration, Pattern, Type. In this dataset they are named Class(target) and then subsequently V2 - V13.

Contact:
```
Muhammad Usman & Adeel Ahmed, usman.madspot '@' gmail.com adeel.ahmed92 '@' gmail.com, Air University, Students at Air University.
```

####2: Attribute Information:

```
Recommendation:0,1 
Style: Bohemia,brief,casual,cute,fashion,flare,novelty,OL,party,sexy,vintage,work. 
Price:Low,Average,Medium,High,Very-High 
Rating:1-5 
Size:S,M,L,XL,Free 
Season:Autumn,winter,Spring,Summer 
NeckLine:O-neck,backless,board-neck,Bowneck,halter,mandarin-collor,open,peterpan-collor,ruffled,scoop,slash-neck,square-collar,sweetheart,turndowncollar,V-neck. 
SleeveLength:full,half,halfsleeves,butterfly,sleveless,short,threequarter,turndown,null 
waiseline:dropped,empire,natural,princess,null. 
Material:wool,cotton,mix etc 
FabricType:shafoon,dobby,popline,satin,knitted,jersey,flannel,corduroy etc 
Decoration:applique,beading,bow,button,cascading,crystal,draped,embroridary,feathers,flowers etc 
Pattern type: solid,animal,dot,leapard etc 
```
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V2 (nominal, 13 distinct): ['Casual', 'Sexy', 'party', 'cute', 'vintage', 'bohemian', 'Brief', 'work', 'Novelty', 'sexy']
V3 (nominal, 8 distinct): ['Average', 'Low', 'low', 'Medium', 'very-high', 'high', 'High']
V4 (numeric, 17 distinct): ['0.0', '4.7', '4.8', '5.0', '4.6', '4.5', '4.4', '4.9', '4.3', '4.0']
V5 (nominal, 7 distinct): ['M', 'free', 'L', 'S', 'XL', 's', 'small']
V6 (nominal, 9 distinct): ['Summer', 'Spring', 'Winter', 'Automn', 'winter', 'Autumn', 'spring', 'summer']
V7 (nominal, 17 distinct): ['o-neck', 'v-neck', 'slash-neck', 'boat-neck', 'Sweetheart', 'turndowncollor', 'bowneck', 'peterpan-collor', 'sqare-collor', 'open']
V8 (nominal, 18 distinct): ['sleevless', 'full', 'short', 'halfsleeve', 'threequarter', 'thressqatar', 'sleeveless', 'sleeevless', 'capsleeves', 'cap-sleeves']
V9 (nominal, 5 distinct): ['natural', 'empire', 'dropped', 'princess']
V10 (nominal, 24 distinct): ['cotton', 'polyster', 'silk', 'chiffonfabric', 'mix', 'rayon', 'nylon', 'milksilk', 'spandex', 'cashmere']
V11 (nominal, 23 distinct): ['chiffon', 'broadcloth', 'worsted', 'jersey', 'shiffon', 'sattin', 'wollen', 'tulle', 'poplin', 'batik']
V12 (nominal, 25 distinct): ['lace', 'sashes', 'beading', 'applique', 'hollowout', 'ruffles', 'bow', 'sequined', 'button', 'pockets']
V13 (nominal, 15 distinct): ['solid', 'print', 'patchwork', 'animal', 'striped', 'dot', 'geometric', 'leopard', 'plaid', 'floral']
'''

CONTEXT = "Dress Sales"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V2", new_name="Style"),
            CuratedFeature(raw_name="V3", new_name="Price"),
            CuratedFeature(raw_name="V4", new_name="Rating"),
            CuratedFeature(raw_name="V5", new_name="Size"),
            CuratedFeature(raw_name="V6", new_name="Season"),
            CuratedFeature(raw_name="V7", new_name="NeckLine"),
            CuratedFeature(raw_name="V8", new_name="SleeveLength"),
            CuratedFeature(raw_name="V9", new_name="Waiseline"),
            CuratedFeature(raw_name="V10", new_name="Material"),
            CuratedFeature(raw_name="V11", new_name="FabricType"),
            CuratedFeature(raw_name="V12", new_name="Decoration"),
            CuratedFeature(raw_name="V13", new_name="Pattern")]