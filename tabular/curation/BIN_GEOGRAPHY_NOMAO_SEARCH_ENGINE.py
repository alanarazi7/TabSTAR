from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: nomao
====
Examples: 34465
====
URL: https://www.openml.org/search?type=data&id=1486
====
Description: **Author**: Nomao Labs

**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Nomao)

**Please cite**: Laurent Candillier and Vincent Lemaire. Design and Analysis of the Nomao Challenge - Active Learning in the Real-World. In: Proceedings of the ALRA : Active Learning in Real-world Applications, Workshop ECML-PKDD 2012, Friday, September 28, 2012, Bristol, UK.

1. Data set title:
Nomao Data Set 


2. Abstract: 
Nomao collects data about places (name, phone, localization...) from many sources. Deduplication consists in detecting what data refer to the same place. Instances in the dataset compare 2 spots.

3. Data Set Characteristics:  

- Univariate
- Area: Computer
- Attribute Characteristics: Real
- Associated Tasks: Classification
- Missing Values?: Yes


4. Source:

(a) Original owner of database (name / phone / snail address / email address) 
Nomao / 00 33 5 62 48 33 90 / 1 avenue Jean Rieux, 31500 Toulouse / challenge '@' nomao.com 
(b) Donor of database (name / phone / snail address / email address) 
Laurent Candillier / - / 1 avenue Jean Rieux, 31500 Toulouse / laurent '@' nomao.com


5. Data Set Information:

The dataset has been enriched during the Nomao Challenge: organized along with the ALRA workshop (Active Learning in Real-world Applications): held at the ECML-PKDD 2012 conference.

5.1. Number of Instances

34,465 instances, mix of continuous and nominal, labeled by human expert.

First 29,104 instances have been labeled with "human prior".
See the corresponding article described in section "3. Past Usage" for more details.

Next 917 instances have been labeled using the active learning method called "marg".
Next 964 instances refer to the active method called "wmarg".
Next 995 instances refer to the active method called "wmarg5".
Next 1,985 instances refer to the active method called "rand" (random selection).

Last instances have been labeled during the corresponding challenge.
More details can be found in http://www.nomao.com/labs/challenge
Next 163 instances refer to the active method called "baseline".
Next 167 instances refer to the active method called "nomao".
And last 170 instances refer to the active method called "tsun".

5.2. Number of Attributes 

120 attributes: 89 continuous, 31 nominal (including the attributes 'label' and 'id'). 

The features are separated by comma.

5.3. Attribute Information: 

Missing data are allowed, represented by question marks '?'.

Labels are +1 if the concerned spots must be merged, -1 if they do not refer to the same entity.

1 id: name is composed of the names of the spots that are compared, separated by a sharp (#).   
2 clean_name_intersect_min: continuous.   
3 clean_name_intersect_max: continuous.   
4 clean_name_levenshtein_sim: continuous.   
5 clean_name_trigram_sim: continuous.   
6 clean_name_levenshtein_term: continuous.   
7 clean_name_trigram_term: continuous.   
8 clean_name_including: n,s,m.   
9 clean_name_equality: n,s,m.   
10 city_intersect_min: continuous.   
11 city_intersect_max: continuous.   
12 city_levenshtein_sim: continuous.   
13 city_trigram_sim: continuous.   
14 city_levenshtein_term: continuous.   
15 city_trigram_term: continuous.   
16 city_including: n,s,m.   
17 city_equality: n,s,m.   
18 zip_intersect_min: continuous.   
19 zip_intersect_max: continuous.   
20 zip_levenshtein_sim: continuous.   
21 zip_trigram_sim: continuous.   
22 zip_levenshtein_term: continuous.   
23 zip_trigram_term: continuous.   
24 zip_including: n,s,m.   
25 zip_equality: n,s,m.   
26 street_intersect_min: continuous.   
27 street_intersect_max: continuous.   
28 street_levenshtein_sim: continuous.   
29 street_trigram_sim: continuous.   
30 street_levenshtein_term: continuous.   
31 street_trigram_term: continuous.   
32 street_including: n,s,m.   
33 street_equality: n,s,m.   
34 website_intersect_min: continuous.   
35 website_intersect_max: continuous.   
36 website_levenshtein_sim: continuous.   
37 website_trigram_sim: continuous.   
38 website_levenshtein_term: continuous.   
39 website_trigram_term: continuous.   
40 website_including: n,s,m.   
41 website_equality: n,s,m.   
42 countryname_intersect_min: continuous.   
43 countryname_intersect_max: continuous.   
44 countryname_levenshtein_sim: continuous.   
45 countryname_trigram_sim: continuous.   
46 countryname_levenshtein_term: continuous.   
47 countryname_trigram_term: continuous.   
48 countryname_including: n,s,m.   
49 countryname_equality: n,s,m.   
50 geocoderlocalityname_intersect_min: continuous.   
51 geocoderlocalityname_intersect_max: continuous.   
52 geocoderlocalityname_levenshtein_sim: continuous.   
53 geocoderlocalityname_trigram_sim: continuous.   
54 geocoderlocalityname_levenshtein_term: continuous.   
55 geocoderlocalityname_trigram_term: continuous.   
56 geocoderlocalityname_including: n,s,m.   
57 geocoderlocalityname_equality: n,s,m.   
58 geocoderinputaddress_intersect_min: continuous.   
59 geocoderinputaddress_intersect_max: continuous.   
60 geocoderinputaddress_levenshtein_sim: continuous.   
61 geocoderinputaddress_trigram_sim: continuous.   
62 geocoderinputaddress_levenshtein_term: continuous.   
63 geocoderinputaddress_trigram_term: continuous.   
64 geocoderinputaddress_including: n,s,m.   
65 geocoderinputaddress_equality: n,s,m.   
66 geocoderoutputaddress_intersect_min: continuous.   
67 geocoderoutputaddress_intersect_max: continuous.   
68 geocoderoutputaddress_levenshtein_sim: continuous.   
69 geocoderoutputaddress_trigram_sim: continuous.   
70 geocoderoutputaddress_levenshtein_term: continuous.   
71 geocoderoutputaddress_trigram_term: continuous.   
72 geocoderoutputaddress_including: n,s,m.   
73 geocoderoutputaddress_equality: n,s,m.   
74 geocoderpostalcodenumber_intersect_min: continuous.   
75 geocoderpostalcodenumber_intersect_max: continuous.   
76 geocoderpostalcodenumber_levenshtein_sim: continuous.   
77 geocoderpostalcodenumber_trigram_sim: continuous.   
78 geocoderpostalcodenumber_levenshtein_term: continuous.   
79 geocoderpostalcodenumber_trigram_term: continuous.   
80 geocoderpostalcodenumber_including: n,s,m.   
81 geocoderpostalcodenumber_equality: n,s,m.   
82 geocodercountrynamecode_intersect_min: continuous.   
83 geocodercountrynamecode_intersect_max: continuous.   
84 geocodercountrynamecode_levenshtein_sim: continuous.   
85 geocodercountrynamecode_trigram_sim: continuous.   
86 geocodercountrynamecode_levenshtein_term: continuous.   
87 geocodercountrynamecode_trigram_term: continuous.   
88 geocodercountrynamecode_including: n,s,m.   
89 geocodercountrynamecode_equality: n,s,m.   
90 phone_diff: continuous.   
91 phone_levenshtein: continuous.   
92 phone_trigram: continuous.   
93 phone_equality: n,s,m.   
94 fax_diff: continuous.   
95 fax_levenshtein: continuous.   
96 fax_trigram: continuous.   
97 fax_equality: n,s,m.   
98 street_number_diff: continuous.   
99 street_number_levenshtein: continuous.   
100 street_number_trigram: continuous.   
101 street_number_equality: n,s,m.   
102 geocode_coordinates_long_diff: continuous.   
103 geocode_coordinates_long_levenshtein: continuous.   
104 geocode_coordinates_long_trigram: continuous.   
105 geocode_coordinates_long_equality: n,s,m.   
106 geocode_coordinates_lat_diff: continuous.   
107 geocode_coordinates_lat_levenshtein: continuous.   
108 geocode_coordinates_lat_trigram: continuous.   
109 geocode_coordinates_lat_equality: n,s,m.   
110 coordinates_long_diff: continuous.   
111 coordinates_long_levenshtein: continuous.   
112 coordinates_long_trigram: continuous.   
113 coordinates_long_equality: n,s,m.   
114 coordinates_lat_diff: continuous.   
115 coordinates_lat_levenshtein: continuous.   
116 coordinates_lat_trigram: continuous.   
117 coordinates_lat_equality: n,s,m.   
118 geocode_coordinates_diff: continuous.   
119 coordinates_diff: continuous.   
120 label: +1,-1.

Relevant Papers: Laurent Candillier and Vincent Lemaire. Design and Analysis of the Nomao Challenge - Active Learning in the Real-World. In: Proceedings of the ALRA : Active Learning in Real-world Applications, Workshop ECML-PKDD 2012, Friday, September 28, 2012, Bristol, UK.
====
Target Variable: Class (nominal, 2 distinct): ['2', '1']
====
Features:

V1 (numeric, 27 distinct): ['1.0', '0.0', '0.5', '0.6667', '0.3333', '0.75', '0.25', '0.8', '0.6', '0.4']
V2 (numeric, 43 distinct): ['0.0', '1.0', '0.5', '0.6667', '0.3333', '0.25', '0.75', '0.4', '0.6', '0.2']
V3 (numeric, 3942 distinct): ['1.0', '0.6667', '0.8', '0.8571', '0.5', '0.5714', '0.75', '0.4', '0.8889', '0.3333']
V4 (numeric, 2207 distinct): ['1.0', '0.0', '0.6667', '0.8', '0.5', '0.8571', '0.5714', '0.4', '0.3333', '0.75']
V5 (numeric, 759 distinct): ['1.0', '0.25', '0.5', '0.2', '0.3333', '0.6667', '0.5714', '0.2143', '0.1429', '0.1667']
V6 (numeric, 929 distinct): ['1.0', '0.0', '0.5', '0.3333', '0.6667', '0.4', '0.7', '0.8', '0.5385', '0.25']
V7 (nominal, 2 distinct): ['2', '1']
V8 (nominal, 2 distinct): ['1', '2']
V9 (numeric, 8 distinct): ['0.8609', '1.0', '0.0', '0.5', '0.6667', '0.75', '0.3333', '0.8']
V10 (numeric, 11 distinct): ['0.8216', '1.0', '0.0', '0.5', '0.6667', '0.75', '0.3333', '0.8', '0.25', '0.6']
V11 (numeric, 240 distinct): ['0.8834', '1.0', '0.6667', '0.5714', '0.0', '0.8333', '0.1', '0.5', '0.1667', '0.125']
V12 (numeric, 136 distinct): ['0.8541', '1.0', '0.6667', '0.0', '0.3636', '0.3333', '0.5', '0.0588', '0.4', '0.4444']
V13 (numeric, 120 distinct): ['0.8696', '1.0', '0.4545', '0.5', '0.5714', '0.0', '0.5556', '0.1', '0.0909', '0.125']
V14 (numeric, 137 distinct): ['0.855', '1.0', '0.0', '0.5', '0.5455', '0.3636', '0.6', '0.6667', '0.0588', '0.4444']
V15 (nominal, 3 distinct): ['1', '3', '2']
V16 (nominal, 3 distinct): ['1', '3', '2']
V17 (numeric, 4 distinct): ['0.7889', '1.0', '0.0', '0.5']
V18 (numeric, 4 distinct): ['0.7832', '1.0', '0.0', '0.5']
V19 (numeric, 20 distinct): ['0.8927', '1.0', '0.8', '0.0', '0.2', '0.6667', '0.25', '0.6', '0.5', '0.8333']
V20 (numeric, 23 distinct): ['0.8338', '1.0', '0.3333', '0.0', '0.5', '0.5714', '0.6667', '0.2', '0.1111', '0.25']
V21 (numeric, 23 distinct): ['0.8903', '1.0', '0.8', '0.0', '0.2', '0.25', '0.6', '0.5', '0.4', '0.4286']
V22 (numeric, 28 distinct): ['0.8306', '1.0', '0.3333', '0.0', '0.5', '0.5714', '0.2', '0.4286', '0.1111', '0.6']
V23 (nominal, 3 distinct): ['1', '3', '2']
V24 (nominal, 3 distinct): ['1', '3', '2']
V25 (numeric, 23 distinct): ['0.6385', '1.0', '0.0', '0.3333', '0.75', '0.25', '0.6667', '0.5', '0.8', '0.2']
V26 (numeric, 43 distinct): ['0.574', '1.0', '0.0', '0.75', '0.5', '0.25', '0.6667', '0.2', '0.8', '0.3333']
V27 (numeric, 2044 distinct): ['0.6889', '1.0', '0.8571', '0.6667', '0.8', '0.75', '0.8889', '0.8333', '0.5', '0.3333']
V28 (numeric, 1082 distinct): ['0.6206', '1.0', '0.0', '0.8571', '0.75', '0.8', '0.6667', '0.2857', '0.8889', '0.25']
V29 (numeric, 541 distinct): ['0.6661', '1.0', '0.25', '0.3333', '0.5', '0.2', '0.8333', '0.8571', '0.2857', '0.1667']
V30 (numeric, 653 distinct): ['0.5873', '1.0', '0.0', '0.8333', '0.0278', '0.6667', '0.75', '0.1667', '0.8', '0.85']
V31 (nominal, 3 distinct): ['1', '2', '3']
V32 (nominal, 3 distinct): ['1', '2', '3']
V33 (numeric, 33 distinct): ['0.8305', '1.0', '0.75', '0.5', '0.6', '0.4', '0.8', '0.0', '0.6667', '0.2']
V34 (numeric, 44 distinct): ['0.7291', '1.0', '0.75', '0.3', '0.6', '0.5', '0.4', '0.8', '0.2222', '0.3333']
V35 (numeric, 356 distinct): ['0.8097', '1.0', '0.75', '0.8571', '0.8', '0.8889', '0.5', '0.8413', '0.7273', '0.7143']
V36 (numeric, 242 distinct): ['0.7785', '1.0', '0.75', '0.4286', '0.6667', '0.8571', '0.5', '0.8', '0.8889', '0.4']
V37 (numeric, 268 distinct): ['0.7529', '1.0', '0.3333', '0.3137', '0.2745', '0.9643', '0.5', '0.4', '0.3871', '0.6']
V38 (numeric, 339 distinct): ['0.7046', '1.0', '0.3488', '0.2063', '0.8387', '0.2419', '0.2', '0.4118', '0.5806', '0.8519']
V39 (nominal, 3 distinct): ['1', '3', '2']
V40 (nominal, 3 distinct): ['1', '2', '3']
V41 (numeric, 3 distinct): ['0.7264', '1.0', '0.0']
V42 (numeric, 3 distinct): ['0.7264', '1.0', '0.0']
V43 (numeric, 34 distinct): ['0.8564', '1.0', '0.7143', '0.5', '0.1667', '0.3333', '0.0', '0.2', '0.1', '0.1111']
V44 (numeric, 27 distinct): ['0.803', '1.0', '0.5', '0.0', '0.2857', '0.0556', '0.25', '0.2667', '0.0741', '0.0526']
V45 (numeric, 28 distinct): ['0.8555', '1.0', '0.7143', '0.5', '0.1667', '0.3333', '0.0', '0.1429', '0.2', '0.2727']
V46 (numeric, 26 distinct): ['0.8029', '1.0', '0.5', '0.0', '0.2857', '0.0556', '0.25', '0.2667', '0.0588', '0.0526']
V47 (nominal, 3 distinct): ['1', '3', '2']
V48 (nominal, 3 distinct): ['1', '3', '2']
V49 (numeric, 7 distinct): ['1.0', '0.0', '0.9525', '0.5', '0.6667', '0.75', '0.3333']
V50 (numeric, 7 distinct): ['1.0', '0.0', '0.9513', '0.3333', '0.5', '0.75', '0.6667']
V51 (numeric, 183 distinct): ['1.0', '0.9647', '0.0', '0.9', '0.3', '0.2857', '0.125', '0.1111', '0.1', '0.5']
V52 (numeric, 92 distinct): ['1.0', '0.0', '0.9573', '0.75', '0.0588', '0.5', '0.1538', '0.6667', '0.0317', '0.0833']
V53 (numeric, 92 distinct): ['1.0', '0.9642', '0.0', '0.9', '0.3', '0.2857', '0.25', '0.1667', '0.1111', '0.2']
V54 (numeric, 90 distinct): ['1.0', '0.0', '0.9571', '0.75', '0.0588', '0.1538', '0.0357', '0.375', '0.0769', '0.05']
V55 (nominal, 3 distinct): ['3', '2', '1']
V56 (nominal, 3 distinct): ['3', '2', '1']
V57 (numeric, 51 distinct): ['1.0', '0.8333', '0.6667', '0.5', '0.8571', '0.8', '0.7143', '0.75', '0.6', '0.3333']
V58 (numeric, 79 distinct): ['1.0', '0.8571', '0.6667', '0.5', '0.7143', '0.8333', '0.5714', '0.75', '0.4286', '0.875']
V59 (numeric, 6861 distinct): ['1.0', '0.9231', '0.8333', '0.9091', '0.8', '0.8571', '0.9333', '0.6667', '0.8889', '0.7273']
V60 (numeric, 4771 distinct): ['1.0', '0.9231', '0.8333', '0.9091', '0.8', '0.8571', '0.6667', '0.9333', '0.7273', '0.875']
V61 (numeric, 1289 distinct): ['1.0', '0.5', '0.6667', '0.6', '0.75', '0.8', '0.625', '0.8333', '0.8889', '0.4545']
V62 (numeric, 1690 distinct): ['1.0', '0.6667', '0.5', '0.75', '0.6', '0.8333', '0.8', '0.9118', '0.8571', '0.8966']
V63 (nominal, 3 distinct): ['2', '3', '1']
V64 (nominal, 3 distinct): ['2', '3', '1']
V65 (numeric, 39 distinct): ['1.0', '0.6667', '0.5', '0.8571', '0.8333', '0.5714', '0.4286', '0.3333', '0.8889', '0.875']
V66 (numeric, 72 distinct): ['1.0', '0.2857', '0.5', '0.8571', '0.4286', '0.3333', '0.375', '0.8333', '0.6667', '0.8889']
V67 (numeric, 3136 distinct): ['1.0', '0.4444', '0.5', '0.6667', '0.6', '0.8571', '0.8', '0.8889', '0.9231', '0.9048']
V68 (numeric, 1852 distinct): ['1.0', '0.5', '0.4444', '0.6667', '0.6', '0.8571', '0.8', '0.875', '0.4', '0.5714']
V69 (numeric, 956 distinct): ['1.0', '0.6667', '0.5641', '0.3333', '0.6', '0.9804', '0.3846', '0.5', '0.7143', '0.375']
V70 (numeric, 1258 distinct): ['1.0', '0.5', '0.4103', '0.9245', '0.3333', '0.3611', '0.3103', '0.8571', '0.325', '0.4']
V71 (nominal, 3 distinct): ['3', '2', '1']
V72 (nominal, 3 distinct): ['3', '2', '1']
V73 (numeric, 4 distinct): ['1.0', '0.9157', '0.0', '0.5']
V74 (numeric, 4 distinct): ['1.0', '0.9133', '0.0', '0.5']
V75 (numeric, 18 distinct): ['1.0', '0.962', '0.8', '0.0', '0.6', '0.5', '0.25', '0.6667', '0.4', '0.2']
V76 (numeric, 19 distinct): ['1.0', '0.9363', '0.3333', '0.5', '0.0', '0.6667', '0.2', '0.1111', '0.1667', '0.2143']
V77 (numeric, 17 distinct): ['1.0', '0.964', '0.8', '0.6', '0.0', '0.25', '0.4', '0.75', '0.6667', '0.5']
V78 (numeric, 21 distinct): ['1.0', '0.9363', '0.3333', '0.0', '0.5', '0.2', '0.6667', '0.1111', '0.4286', '0.5556']
V79 (nominal, 3 distinct): ['3', '1', '2']
V80 (nominal, 3 distinct): ['3', '1', '2']
V81 (numeric, 3 distinct): ['1.0', '0.9998', '0.0']
V82 (numeric, 3 distinct): ['1.0', '0.9998', '0.0']
V83 (numeric, 3 distinct): ['1.0', '0.9998', '0.0']
V84 (numeric, 3 distinct): ['1.0', '0.9998', '0.0']
V85 (numeric, 3 distinct): ['1.0', '0.9998', '0.0']
V86 (numeric, 3 distinct): ['1.0', '0.9998', '0.0']
V87 (nominal, 3 distinct): ['3', '1', '2']
V88 (nominal, 3 distinct): ['3', '1', '2']
V89 (numeric, 744 distinct): ['0.6625', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0217', '0.0169', '0.0', '0.0']
V90 (numeric, 29 distinct): ['0.8423', '1.0', '0.4545', '0.5455', '0.8182', '0.6364', '0.3636', '0.5', '0.9091', '0.6']
V91 (numeric, 62 distinct): ['0.7646', '1.0', '0.2632', '0.6', '0.0909', '0.0476', '0.2', '0.4118', '0.2222', '0.375']
V92 (nominal, 3 distinct): ['1', '3', '2']
V93 (numeric, 140 distinct): ['0.5301', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0058', '0.5']
V94 (numeric, 20 distinct): ['0.7608', '1.0', '0.5', '0.4', '0.6', '0.5455', '0.4545', '0.3', '0.7', '0.7273']
V95 (numeric, 31 distinct): ['0.6472', '1.0', '0.2222', '0.0476', '0.2941', '0.2', '0.375', '0.2632', '0.1579', '0.4118']
V96 (nominal, 3 distinct): ['1', '3', '2']
V97 (numeric, 299 distinct): ['1.0', '0.7912', '0.3333', '0.5', '0.1429', '0.25', '0.2', '0.0', '0.1111', '0.0909']
V98 (numeric, 18 distinct): ['1.0', '0.8882', '0.5', '0.75', '0.6667', '0.4', '0.0', '0.6', '0.4286', '0.8']
V99 (numeric, 26 distinct): ['1.0', '0.829', '0.25', '0.2857', '0.3333', '0.4286', '0.0', '0.2222', '0.5', '0.1818']
V100 (nominal, 3 distinct): ['3', '2', '1']
V101 (numeric, 5802 distinct): ['1.0', '0.9998', '1.0', '1.0', '1.0', '0.9999', '0.9999', '0.9997', '0.9998', '0.9999']
V102 (numeric, 46 distinct): ['1.0', '0.5556', '0.4444', '0.6667', '0.3333', '0.5', '0.7778', '0.6', '0.7', '0.2222']
V103 (numeric, 79 distinct): ['1.0', '0.25', '0.1765', '0.3333', '0.4286', '0.2667', '0.2857', '0.1875', '0.3571', '0.1111']
V104 (nominal, 3 distinct): ['2', '3', '1']
V105 (numeric, 5461 distinct): ['1.0', '1.0', '0.9999', '0.9999', '0.9999', '0.9999', '0.9999', '0.9999', '1.0', '0.9999']
V106 (numeric, 32 distinct): ['1.0', '0.6', '0.5', '0.7', '0.4', '0.8', '0.3', '0.9', '0.5556', '0.6667']
V107 (numeric, 85 distinct): ['1.0', '0.375', '0.2941', '0.2222', '0.4667', '0.2353', '0.4', '0.3125', '0.1579', '0.25']
V108 (nominal, 3 distinct): ['2', '3', '1']
V109 (numeric, 5095 distinct): ['0.9787', '1.0', '1.0', '1.0', '1.0', '1.0', '0.999', '1.0', '1.0', '1.0']
V110 (numeric, 67 distinct): ['0.6079', '0.5', '1.0', '0.625', '0.5556', '0.6667', '0.75', '0.4444', '0.375', '0.7778']
V111 (numeric, 102 distinct): ['0.3962', '1.0', '0.2857', '0.2', '0.3846', '0.3571', '0.5', '0.3077', '0.2667', '0.3333']
V112 (nominal, 3 distinct): ['1', '2', '3']
V113 (numeric, 4687 distinct): ['0.9826', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0']
V114 (numeric, 56 distinct): ['0.6443', '0.6667', '0.5556', '1.0', '0.7778', '0.7', '0.4444', '0.6', '0.5', '0.5385']
V115 (numeric, 104 distinct): ['0.4381', '0.3333', '1.0', '0.25', '0.4286', '0.5', '0.5385', '0.4', '0.3571', '0.3125']
V116 (nominal, 3 distinct): ['1', '2', '3']
V117 (numeric, 2039 distinct): ['1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0']
V118 (numeric, 1726 distinct): ['0.9793', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0']
'''

CONTEXT = "Nomao Search Engine"
TARGET = CuratedTarget(raw_name="Class", new_name="Place should be merged", task_type=SupervisedTask.BINARY,
                       label_mapping={'1': 'Yes', '2': 'No'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Clean Name Intersect Min"),
            CuratedFeature(raw_name="V2", new_name="Clean Name Intersect Max"),
            CuratedFeature(raw_name="V3", new_name="Clean Name Levenshtein Sim"),
            CuratedFeature(raw_name="V4", new_name="Clean Name Trigram Sim"),
            CuratedFeature(raw_name="V5", new_name="Clean Name Levenshtein Term"),
            CuratedFeature(raw_name="V6", new_name="Clean Name Trigram Term"),
            CuratedFeature(raw_name="V7", new_name="Clean Name Including"),
            CuratedFeature(raw_name="V8", new_name="Clean Name Equality"),
            CuratedFeature(raw_name="V9", new_name="City Intersect Min"),
            CuratedFeature(raw_name="V10", new_name="City Intersect Max"),
            CuratedFeature(raw_name="V11", new_name="City Levenshtein Sim"),
            CuratedFeature(raw_name="V12", new_name="City Trigram Sim"),
            CuratedFeature(raw_name="V13", new_name="City Levenshtein Term"),
            CuratedFeature(raw_name="V14", new_name="City Trigram Term"),
            CuratedFeature(raw_name="V15", new_name="City Including"),
            CuratedFeature(raw_name="V16", new_name="City Equality"),
            CuratedFeature(raw_name="V17", new_name="Zip Intersect Min"),
            CuratedFeature(raw_name="V18", new_name="Zip Intersect Max"),
            CuratedFeature(raw_name="V19", new_name="Zip Levenshtein Sim"),
            CuratedFeature(raw_name="V20", new_name="Zip Trigram Sim"),
            CuratedFeature(raw_name="V21", new_name="Zip Levenshtein Term"),
            CuratedFeature(raw_name="V22", new_name="Zip Trigram Term"),
            CuratedFeature(raw_name="V23", new_name="Zip Including"),
            CuratedFeature(raw_name="V24", new_name="Zip Equality"),
            CuratedFeature(raw_name="V25", new_name="Street Intersect Min"),
            CuratedFeature(raw_name="V26", new_name="Street Intersect Max"),
            CuratedFeature(raw_name="V27", new_name="Street Levenshtein Sim"),
            CuratedFeature(raw_name="V28", new_name="Street Trigram Sim"),
            CuratedFeature(raw_name="V29", new_name="Street Levenshtein Term"),
            CuratedFeature(raw_name="V30", new_name="Street Trigram Term"),
            CuratedFeature(raw_name="V31", new_name="Street Including"),
            CuratedFeature(raw_name="V32", new_name="Street Equality"),
            CuratedFeature(raw_name="V33", new_name="Website Intersect Min"),
            CuratedFeature(raw_name="V34", new_name="Website Intersect Max"),
            CuratedFeature(raw_name="V35", new_name="Website Levenshtein Sim"),
            CuratedFeature(raw_name="V36", new_name="Website Trigram Sim"),
            CuratedFeature(raw_name="V37", new_name="Website Levenshtein Term"),
            CuratedFeature(raw_name="V38", new_name="Website Trigram Term"),
            CuratedFeature(raw_name="V39", new_name="Website Including"),
            CuratedFeature(raw_name="V40", new_name="Website Equality"),
            CuratedFeature(raw_name="V41", new_name="Country Name Intersect Min"),
            CuratedFeature(raw_name="V42", new_name="Country Name Intersect Max"),
            CuratedFeature(raw_name="V43", new_name="Country Name Levenshtein Sim"),
            CuratedFeature(raw_name="V44", new_name="Country Name Trigram Sim"),
            CuratedFeature(raw_name="V45", new_name="Country Name Levenshtein Term"),
            CuratedFeature(raw_name="V46", new_name="Country Name Trigram Term"),
            CuratedFeature(raw_name="V47", new_name="Country Name Including"),
            CuratedFeature(raw_name="V48", new_name="Country Name Equality"),
            CuratedFeature(raw_name="V49", new_name="Geocoder Locality Name Intersect Min"),
            CuratedFeature(raw_name="V50", new_name="Geocoder Locality Name Intersect Max"),
            CuratedFeature(raw_name="V51", new_name="Geocoder Locality Name Levenshtein Sim"),
            CuratedFeature(raw_name="V52", new_name="Geocoder Locality Name Trigram Sim"),
            CuratedFeature(raw_name="V53", new_name="Geocoder Locality Name Levenshtein Term"),
            CuratedFeature(raw_name="V54", new_name="Geocoder Locality Name Trigram Term"),
            CuratedFeature(raw_name="V55", new_name="Geocoder Locality Name Including"),
            CuratedFeature(raw_name="V56", new_name="Geocoder Locality Name Equality"),
            CuratedFeature(raw_name="V57", new_name="Geocoder Input Address Intersect Min"),
            CuratedFeature(raw_name="V58", new_name="Geocoder Input Address Intersect Max"),
            CuratedFeature(raw_name="V59", new_name="Geocoder Input Address Levenshtein Sim"),
            CuratedFeature(raw_name="V60", new_name="Geocoder Input Address Trigram Sim"),
            CuratedFeature(raw_name="V61", new_name="Geocoder Input Address Levenshtein Term"),
            CuratedFeature(raw_name="V62", new_name="Geocoder Input Address Trigram Term"),
            CuratedFeature(raw_name="V63", new_name="Geocoder Input Address Including"),
            CuratedFeature(raw_name="V64", new_name="Geocoder Input Address Equality"),
            CuratedFeature(raw_name="V65", new_name="Geocoder Output Address Intersect Min"),
            CuratedFeature(raw_name="V66", new_name="Geocoder Output Address Intersect Max"),
            CuratedFeature(raw_name="V67", new_name="Geocoder Output Address Levenshtein Sim"),
            CuratedFeature(raw_name="V68", new_name="Geocoder Output Address Trigram Sim"),
            CuratedFeature(raw_name="V69", new_name="Geocoder Output Address Levenshtein Term"),
            CuratedFeature(raw_name="V70", new_name="Geocoder Output Address Trigram Term"),
            CuratedFeature(raw_name="V71", new_name="Geocoder Output Address Including"),
            CuratedFeature(raw_name="V72", new_name="Geocoder Output Address Equality"),
            CuratedFeature(raw_name="V73", new_name="Geocoder Postal Code Number Intersect Min"),
            CuratedFeature(raw_name="V74", new_name="Geocoder Postal Code Number Intersect Max"),
            CuratedFeature(raw_name="V75", new_name="Geocoder Postal Code Number Levenshtein Sim"),
            CuratedFeature(raw_name="V76", new_name="Geocoder Postal Code Number Trigram Sim"),
            CuratedFeature(raw_name="V77", new_name="Geocoder Postal Code Number Levenshtein Term"),
            CuratedFeature(raw_name="V78", new_name="Geocoder Postal Code Number Trigram Term"),
            CuratedFeature(raw_name="V79", new_name="Geocoder Postal Code Number Including"),
            CuratedFeature(raw_name="V80", new_name="Geocoder Postal Code Number Equality"),
            CuratedFeature(raw_name="V81", new_name="Geocoder Country Name Code Intersect Min"),
            CuratedFeature(raw_name="V82", new_name="Geocoder Country Name Code Intersect Max"),
            CuratedFeature(raw_name="V83", new_name="Geocoder Country Name Code Levenshtein Sim"),
            CuratedFeature(raw_name="V84", new_name="Geocoder Country Name Code Trigram Sim"),
            CuratedFeature(raw_name="V85", new_name="Geocoder Country Name Code Levenshtein Term"),
            CuratedFeature(raw_name="V86", new_name="Geocoder Country Name Code Trigram Term"),
            CuratedFeature(raw_name="V87", new_name="Geocoder Country Name Code Including"),
            CuratedFeature(raw_name="V88", new_name="Geocoder Country Name Code Equality"),
            CuratedFeature(raw_name="V89", new_name="Phone Diff"),
            CuratedFeature(raw_name="V90", new_name="Phone Levenshtein"),
            CuratedFeature(raw_name="V91", new_name="Phone Trigram"),
            CuratedFeature(raw_name="V92", new_name="Phone Equality"),
            CuratedFeature(raw_name="V93", new_name="Fax Diff"),
            CuratedFeature(raw_name="V94", new_name="Fax Levenshtein"),
            CuratedFeature(raw_name="V95", new_name="Fax Trigram"),
            CuratedFeature(raw_name="V96", new_name="Fax Equality"),
            CuratedFeature(raw_name="V97", new_name="Street Number Diff"),
            CuratedFeature(raw_name="V98", new_name="Street Number Levenshtein"),
            CuratedFeature(raw_name="V99", new_name="Street Number Trigram"),
            CuratedFeature(raw_name="V100", new_name="Street Number Equality"),
            CuratedFeature(raw_name="V101", new_name="Geocode Coordinates Long Diff"),
            CuratedFeature(raw_name="V102", new_name="Geocode Coordinates Long Levenshtein"),
            CuratedFeature(raw_name="V103", new_name="Geocode Coordinates Long Trigram"),
            CuratedFeature(raw_name="V104", new_name="Geocode Coordinates Long Equality"),
            CuratedFeature(raw_name="V105", new_name="Geocode Coordinates Lat Diff"),
            CuratedFeature(raw_name="V106", new_name="Geocode Coordinates Lat Levenshtein"),
            CuratedFeature(raw_name="V107", new_name="Geocode Coordinates Lat Trigram"),
            CuratedFeature(raw_name="V108", new_name="Geocode Coordinates Lat Equality"),
            CuratedFeature(raw_name="V109", new_name="Coordinates Long Diff"),
            CuratedFeature(raw_name="V110", new_name="Coordinates Long Levenshtein"),
            CuratedFeature(raw_name="V111", new_name="Coordinates Long Trigram"),
            CuratedFeature(raw_name="V112", new_name="Coordinates Long Equality"),
            CuratedFeature(raw_name="V113", new_name="Coordinates Lat Diff"),
            CuratedFeature(raw_name="V114", new_name="Coordinates Lat Levenshtein"),
            CuratedFeature(raw_name="V115", new_name="Coordinates Lat Trigram"),
            CuratedFeature(raw_name="V116", new_name="Coordinates Lat Equality"),
            CuratedFeature(raw_name="V117", new_name="Geocode Coordinates Diff"),
            CuratedFeature(raw_name="V118", new_name="Coordinates Diff")]
