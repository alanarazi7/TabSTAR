from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Buzzinsocialmedia_Twitter
====
Examples: 583250
====
URL: https://www.openml.org/search?type=data&id=4549
====
Description: **Author**: Creators :  François Kawala (1","2) Ahlame Douzal (1) Eric Gaussier (1) Eustache Diemert (2) Institutions :  (1) Université Joseph Fourier (Grenoble I) Laboratoire d'informatique de Grenoble (LIG) (2) BestofMedia Group Donor:  BestofMedia (ediemert '@' bestofmedia.com)  
**Source**: UCI 
**Please cite**: Pr&eacute;dictions d&rsquo;activit&eacute; dans les r&eacute;seaux sociaux en ligne (F. Kawala, A. Douzal-Chouakria, E. Gaussier, E. Dimert), In Actes de la Conf&eacute;rence sur les Mod&egrave;les et l&prime;Analyse des R&eacute;seaux : Approches Math&eacute;matiques et Informatique (MARAMI), pp. 16, 2013.  

Abstract: This data-set contains examples of buzz events from two different social networks: Twitter, and Tom's Hardware, a forum network focusing on new technology with more conservative dynamics.
Source:

Creators : 
Fran&ccedil;ois Kawala (1,2) Ahlame Douzal (1) Eric Gaussier (1) Eustache Diemert (2)
Institutions : 
(1) Universit&eacute; Joseph Fourier (Grenoble I)
Laboratoire d'informatique de Grenoble (LIG)
(2) BestofMedia Group
Donor: 
BestofMedia (ediemert '@' bestofmedia.com)


Data Set Information:

Please see [Web Link]


Attribute Information:

Please see [Web Link]


Relevant Papers:

Pr&eacute;dictions d&rsquo;activit&eacute; dans les r&eacute;seaux sociaux en ligne (F. Kawala, A. Douzal-Chouakria, E. Gaussier, E. Dimert), In Actes de la Conf&eacute;rence sur les Mod&egrave;les et l&prime;Analyse des R&eacute;seaux : Approches Math&eacute;matiques et Informatique (MARAMI), pp. 16, 2013.



Citation Request:

Pr&eacute;dictions d&rsquo;activit&eacute; dans les r&eacute;seaux sociaux en ligne (F. Kawala, A. Douzal-Chouakria, E. Gaussier, E. Dimert), In Actes de la Conf&eacute;rence sur les Mod&egrave;les et l&prime;Analyse des R&eacute;seaux : Approches Math&eacute;matiques et Informatique (MARAMI), pp. 16, 2013.
====
Target Variable: Annotation (numeric, 8123 distinct): ['0.5', '1.0', '0.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']
====
Features:

NCD_0 (numeric, 4410 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NCD_1 (numeric, 4359 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NCD_2 (numeric, 4762 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NCD_3 (numeric, 5162 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NCD_4 (numeric, 5435 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NCD_5 (numeric, 5656 distinct): ['1.0', '2.0', '3.0', '0.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NCD_6 (numeric, 5656 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_0 (numeric, 2505 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_1 (numeric, 2551 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_2 (numeric, 2797 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_3 (numeric, 3009 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_4 (numeric, 3176 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_5 (numeric, 3299 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
AI_6 (numeric, 3299 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
ASNA_0 (numeric, 5408 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNA_1 (numeric, 5344 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNA_2 (numeric, 5604 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNA_3 (numeric, 5811 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNA_4 (numeric, 6031 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNA_5 (numeric, 6242 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNA_6 (numeric, 6150 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
BL_0 (numeric, 9090 distinct): ['1.0', '0.0', '0.5', '0.75', '0.8', '0.6667', '0.9091', '0.875', '0.8333', '0.8889']
BL_1 (numeric, 8621 distinct): ['1.0', '0.0', '0.5', '0.75', '0.8', '0.6667', '0.9091', '0.875', '0.9333', '0.8889']
BL_2 (numeric, 8839 distinct): ['1.0', '0.0', '0.8', '0.5', '0.9231', '0.9091', '0.75', '0.8889', '0.6667', '0.875']
BL_3 (numeric, 9349 distinct): ['1.0', '0.0', '0.75', '0.8', '0.5', '0.8333', '0.8889', '0.6667', '0.9231', '0.875']
BL_4 (numeric, 9708 distinct): ['1.0', '0.0', '0.8', '0.5', '0.8333', '0.6667', '0.75', '0.875', '0.8571', '0.8889']
BL_5 (numeric, 9986 distinct): ['1.0', '0.0', '0.8', '0.75', '0.6667', '0.5', '0.8333', '0.875', '0.8571', '0.9']
BL_6 (numeric, 10077 distinct): ['1.0', '0.0', '0.75', '0.5', '0.8', '0.6667', '0.8333', '0.875', '0.8571', '0.9']
NAC_0 (numeric, 4562 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAC_1 (numeric, 4492 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAC_2 (numeric, 4896 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAC_3 (numeric, 5259 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAC_4 (numeric, 5577 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAC_5 (numeric, 5812 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAC_6 (numeric, 5837 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
ASNAC_0 (numeric, 3815 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNAC_1 (numeric, 3734 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNAC_2 (numeric, 3957 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNAC_3 (numeric, 4125 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNAC_4 (numeric, 4249 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNAC_5 (numeric, 4376 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
ASNAC_6 (numeric, 4340 distinct): ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
CS_0 (numeric, 2 distinct): ['1', '0']
CS_1 (numeric, 2 distinct): ['1', '0']
CS_2 (numeric, 2 distinct): ['1', '0']
CS_3 (numeric, 2 distinct): ['1', '0']
CS_4 (numeric, 2 distinct): ['1', '0']
CS_5 (numeric, 2 distinct): ['1', '0']
CS_6 (numeric, 2 distinct): ['1', '0']
AT_0 (numeric, 37782 distinct): ['1.0', '0.0', '1.5', '2.0', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
AT_1 (numeric, 36415 distinct): ['1.0', '0.0', '1.5', '2.0', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
AT_2 (numeric, 39407 distinct): ['1.0', '0.0', '1.5', '1.3333', '1.25', '2.0', '1.2', '1.1667', '1.1429', '1.125']
AT_3 (numeric, 42497 distinct): ['1.0', '0.0', '1.5', '1.3333', '1.25', '2.0', '1.2', '1.1667', '1.1429', '1.125']
AT_4 (numeric, 45177 distinct): ['1.0', '0.0', '1.5', '1.3333', '2.0', '1.25', '1.2', '1.1667', '1.1429', '1.125']
AT_5 (numeric, 47746 distinct): ['1.0', '0.0', '1.5', '1.3333', '2.0', '1.25', '1.2', '1.1667', '1.1429', '1.125']
AT_6 (numeric, 47608 distinct): ['1.0', '0.0', '1.5', '1.3333', '2.0', '1.25', '1.2', '1.1667', '1.1429', '1.125']
NA_0 (numeric, 3847 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NA_1 (numeric, 3758 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NA_2 (numeric, 4144 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NA_3 (numeric, 4451 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NA_4 (numeric, 4723 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NA_5 (numeric, 4930 distinct): ['1.0', '2.0', '3.0', '0.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NA_6 (numeric, 4889 distinct): ['1.0', '2.0', '3.0', '0.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
ADL_0 (numeric, 43341 distinct): ['1.0', '0.0', '2.0', '1.5', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
ADL_1 (numeric, 41808 distinct): ['1.0', '0.0', '2.0', '1.5', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
ADL_2 (numeric, 45332 distinct): ['1.0', '0.0', '1.5', '2.0', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
ADL_3 (numeric, 48872 distinct): ['1.0', '0.0', '2.0', '1.5', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
ADL_4 (numeric, 52049 distinct): ['1.0', '0.0', '2.0', '1.5', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
ADL_5 (numeric, 54881 distinct): ['1.0', '0.0', '2.0', '1.5', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
ADL_6 (numeric, 54668 distinct): ['1.0', '0.0', '2.0', '1.5', '1.3333', '1.25', '1.2', '1.1667', '1.1429', '1.125']
NAD_0 (numeric, 4417 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAD_1 (numeric, 4373 distinct): ['1.0', '0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAD_2 (numeric, 4740 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAD_3 (numeric, 5123 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAD_4 (numeric, 5412 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAD_5 (numeric, 5671 distinct): ['1.0', '2.0', '3.0', '0.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NAD_6 (numeric, 5682 distinct): ['1.0', '2.0', '3.0', '0.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
'''

CONTEXT = "Buzz in Social Media including Twitter"
TARGET = CuratedTarget(raw_name="Annotation", new_name="Buzz in Social Media", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []