from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mushroom
====
Examples: 8124
====
URL: https://www.openml.org/search?type=data&id=24
====
Description: **Author**: [Jeff Schlimmer](Jeffrey.Schlimmer@a.gp.cs.cmu.edu)  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/mushroom) - 1981     
**Please cite**:  The Audubon Society Field Guide to North American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A. Knopf 


### Description

This dataset describes mushrooms in terms of their physical characteristics. They are classified into: poisonous or edible.

### Source
```
(a) Origin: 
Mushroom records are drawn from The Audubon Society Field Guide to North American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A. Knopf 

(b) Donor: 
Jeff Schlimmer (Jeffrey.Schlimmer '@' a.gp.cs.cmu.edu)
```

### Dataset description

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

### Attributes Information
```
1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
4. bruises?: bruises=t,no=f 
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
6. gill-attachment: attached=a,descending=d,free=f,notched=n 
7. gill-spacing: close=c,crowded=w,distant=d 
8. gill-size: broad=b,narrow=n 
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
10. stalk-shape: enlarging=e,tapering=t 
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
16. veil-type: partial=p,universal=u 
17. veil-color: brown=n,orange=o,white=w,yellow=y 
18. ring-number: none=n,one=o,two=t 
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
```

### Relevant papers

Schlimmer,J.S. (1987). Concept Acquisition Through Representational Adjustment (Technical Report 87-19). Doctoral disseration, Department of Information and Computer Science, University of California, Irvine. 

Iba,W., Wogulis,J., & Langley,P. (1988). Trading off Simplicity and Coverage in Incremental Concept Learning. In Proceedings of the 5th International Conference on Machine Learning, 73-79. Ann Arbor, Michigan: Morgan Kaufmann. 

Duch W, Adamczak R, Grabczewski K (1996) Extraction of logical rules from training data using backpropagation networks, in: Proc. of the The 1st Online Workshop on Soft Computing, 19-30.Aug.1996, pp. 25-30, [Web Link] 

Duch W, Adamczak R, Grabczewski K, Ishikawa M, Ueda H, Extraction of crisp logical rules using constrained backpropagation networks - comparison of two new approaches, in: Proc. of the European Symposium on Artificial Neural Networks (ESANN'97), Bruge, Belgium 16-18.4.1997.
====
Target Variable: class (nominal, 2 distinct): ['e', 'p']
====
Features:

cap-shape (nominal, 6 distinct): ['x', 'f', 'k', 'b', 's', 'c']
cap-surface (nominal, 4 distinct): ['y', 's', 'f', 'g']
cap-color (nominal, 10 distinct): ['n', 'g', 'e', 'y', 'w', 'b', 'p', 'c', 'r', 'u']
bruises%3F (nominal, 2 distinct): ['f', 't']
odor (nominal, 9 distinct): ['n', 'f', 's', 'y', 'a', 'l', 'p', 'c', 'm']
gill-attachment (nominal, 2 distinct): ['f', 'a', 'd', 'n']
gill-spacing (nominal, 2 distinct): ['c', 'w', 'd']
gill-size (nominal, 2 distinct): ['b', 'n']
gill-color (nominal, 12 distinct): ['b', 'p', 'w', 'n', 'g', 'h', 'u', 'k', 'e', 'y']
stalk-shape (nominal, 2 distinct): ['t', 'e']
stalk-root (nominal, 5 distinct): ['b', 'e', 'c', 'r', 'u', 'z']
stalk-surface-above-ring (nominal, 4 distinct): ['s', 'k', 'f', 'y']
stalk-surface-below-ring (nominal, 4 distinct): ['s', 'k', 'f', 'y']
stalk-color-above-ring (nominal, 9 distinct): ['w', 'p', 'g', 'n', 'b', 'o', 'e', 'c', 'y']
stalk-color-below-ring (nominal, 9 distinct): ['w', 'p', 'g', 'n', 'b', 'o', 'e', 'c', 'y']
veil-type (nominal, 1 distinct): ['p', 'u']
veil-color (nominal, 4 distinct): ['w', 'n', 'o', 'y']
ring-number (nominal, 3 distinct): ['o', 't', 'n']
ring-type (nominal, 5 distinct): ['p', 'e', 'l', 'f', 'n', 'c', 's', 'z']
spore-print-color (nominal, 9 distinct): ['w', 'n', 'k', 'h', 'r', 'b', 'o', 'u', 'y']
population (nominal, 6 distinct): ['v', 'y', 's', 'n', 'a', 'c']
habitat (nominal, 7 distinct): ['d', 'g', 'p', 'l', 'u', 'm', 'w']
'''

CONTEXT = "Mushroom Physical Characteristics and Poisonous Detection"
TARGET = CuratedTarget(raw_name="class", new_name="Poisonous", task_type=SupervisedTask.BINARY,
                       label_mapping={"e": "edible", "p": "poisonous"})
COLS_TO_DROP = ['veil-type', # Single value
                ]
FEATURES = [CuratedFeature(raw_name="cap-shape",
                           value_mapping={"x": "convex", "f": "flat", "k": "knobbed", "b": "bell", "s": "sunken", "c": "conical"}),
            CuratedFeature(raw_name="cap-surface",
                           value_mapping={"y": "fibrous", "s": "smooth", "f": "scaly", "g": "grooves"}),
            CuratedFeature(raw_name="cap-color",
                           value_mapping={"n": "brown", "g": "gray", "e": "red", "y": "yellow", "w": "white", "b": "buff", "p": "pink", "c": "cinnamon", "r": "green", "u": "purple"}),
            CuratedFeature(raw_name="bruises%3F",
                           value_mapping={"f": "no", "t": "yes"}),
            CuratedFeature(raw_name="odor",
                           value_mapping={"n": "none", "f": "foul", "s": "spicy", "y": "fishy", "a": "almond", "l": "anise", "p": "pungent", "c": "creosote", "m": "musty"}),
            CuratedFeature(raw_name="gill-attachment",
                           value_mapping={"f": "free", "a": "attached", "d": "descending", "n": "notched"},
                           allow_missing_key=True),
            CuratedFeature(raw_name="gill-spacing",
                           value_mapping={"c": "close", "w": "crowded", "d": "distant"},
                           allow_missing_key=True),
            CuratedFeature(raw_name="gill-size",
                           value_mapping={"b": "broad", "n": "narrow"}),
            CuratedFeature(raw_name="gill-color",
                           value_mapping={"b": "buff", "p": "pink", "w": "white", "n": "brown", "g": "gray", "h": "chocolate", "u": "purple", "k": "black", "e": "red", "y": "yellow"}),
            CuratedFeature(raw_name="stalk-shape",
                           value_mapping={"t": "tapering", "e": "enlarging"}),
            CuratedFeature(raw_name="stalk-root",
                           value_mapping={"b": "bulbous", "e": "equal", "c": "club", "r": "rooted"}),
            CuratedFeature(raw_name="stalk-surface-above-ring",
                           value_mapping={"s": "smooth", "k": "silky", "f": "fibrous", "y": "scaly"}),
            CuratedFeature(raw_name="stalk-surface-below-ring",
                           value_mapping={"s": "smooth", "k": "silky", "f": "fibrous", "y": "scaly"}),
            CuratedFeature(raw_name="stalk-color-above-ring",
                           value_mapping={"w": "white", "p": "pink", "g": "gray", "n": "brown", "b": "buff", "o": "orange", "e": "red", "c": "cinnamon", "y": "yellow"}),
            CuratedFeature(raw_name="stalk-color-below-ring",
                           value_mapping={"w": "white", "p": "pink", "g": "gray", "n": "brown", "b": "buff", "o": "orange", "e": "red", "c": "cinnamon", "y": "yellow"}),
            CuratedFeature(raw_name="veil-color",
                           value_mapping={"w": "white", "n": "brown", "o": "orange", "y": "yellow"}),
            CuratedFeature(raw_name="ring-number",
                           value_mapping={"o": "none", "t": "two", "n": "one"}),
            CuratedFeature(raw_name="ring-type",
                           value_mapping={"p": "pendant", "e": "evanescent", "l": "large", "f": "flaring", "n": "none"}),
            CuratedFeature(raw_name="spore-print-color",
                           value_mapping={"w": "white", "n": "brown", "k": "black", "h": "chocolate", "r": "green", "b": "buff", "o": "orange", "u": "purple", "y": "yellow"}),
            CuratedFeature(raw_name="population",
                           value_mapping={"v": "several", "y": "solitary", "s": "scattered", "n": "numerous", "a": "abundant", "c": "clustered"}),
            CuratedFeature(raw_name="habitat",
                           value_mapping={"d": "woods", "g": "grasses", "p": "paths", "l": "leaves", "u": "urban", "m": "meadows", "w": "waste"})]