from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wholesale-customers
====
Examples: 440
====
URL: https://www.openml.org/search?type=data&id=1511
====
Description: **Author**: Margarida G. M. S. Cardoso      
**Source**: UCI     
**Please cite**: Abreu, N. (2011). Analise do perfil do cliente Recheio e desenvolvimento de um sistema promocional. Mestrado em Marketing, ISCTE-IUL, Lisbon.  

* Title:   
Wholesale customers Data Set 

* Abstract:   
The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories

* Source:  
Margarida G. M. S. Cardoso, margarida.cardoso '@' iscte.pt, ISCTE-IUL, Lisbon, Portugal

* Attribute Information:

1) FRESH: annual spending (m.u.) on fresh products (Continuous); 
2) MILK: annual spending (m.u.) on milk products (Continuous); 
3) GROCERY: annual spending (m.u.)on grocery products (Continuous); 
4) FROZEN: annual spending (m.u.)on frozen products (Continuous) 
5) DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous) 
6) DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous); 
7) CHANNEL: customers' Channel - Horeca (Hotel/Restaurant/Café) or Retail channel (Nominal) 
8) REGION: customers' Region - Lisbon, Porto or Other (Nominal) 

Descriptive Statistics: 

(Minimum, Maximum, Mean, Std. Deviation) 
FRESH ( 3, 112151, 12000.30, 12647.329) 
MILK (55, 73498, 5796.27, 7380.377) 
GROCERY (3, 92780, 7951.28, 9503.163) 
FROZEN (25, 60869, 3071.93, 4854.673) 
DETERGENTS_PAPER (3, 40827, 2881.49, 4767.854) 
DELICATESSEN (3, 47943, 1524.87, 2820.106) 

REGION Frequency 
Lisbon 77 
Oporto 47 
Other Region 316 
Total 440 

CHANNEL Frequency 
Horeca 298 
Retail 142 
Total 440 


* Relevant Papers:

Cardoso, Margarida G.M.S. (2013). Logical discriminant models â€“ Chapter 8 in Quantitative Modeling in Marketing and Management Edited by Luiz Moutinho and Kun-Huang Huarng. World Scientific. p. 223-253. ISBN 978-9814407717 

Jean-Patrick Baudry, Margarida Cardoso, Gilles Celeux, Maria JosÃ© Amorim, Ana Sousa Ferreira (2012). Enhancing the selection of a model-based clustering with external qualitative variables. RESEARCH REPORT NÂ° 8124, October 2012, Project-Team SELECT. INRIA Saclay - ÃŽle-de-France, Projet select, UniversitÃ© Paris-Sud 11
====
Target Variable: Channel (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 3 distinct): ['3', '1', '2']
V2 (numeric, 433 distinct): ['9670.0', '3.0', '18044.0', '8040.0', '514.0', '3366.0', '7149.0', '5283.0', '16448.0', '444.0']
V3 (numeric, 421 distinct): ['3045.0', '1610.0', '5139.0', '2428.0', '3587.0', '1032.0', '1897.0', '829.0', '4230.0', '3880.0']
V4 (numeric, 430 distinct): ['1664.0', '2062.0', '683.0', '3600.0', '6536.0', '2406.0', '10391.0', '2405.0', '1493.0', '1563.0']
V5 (numeric, 426 distinct): ['2540.0', '425.0', '1285.0', '4324.0', '1619.0', '779.0', '937.0', '402.0', '364.0', '744.0']
V6 (numeric, 417 distinct): ['118.0', '955.0', '256.0', '69.0', '918.0', '483.0', '212.0', '3.0', '93.0', '210.0']
V7 (numeric, 403 distinct): ['3.0', '834.0', '548.0', '610.0', '395.0', '1215.0', '46.0', '436.0', '1117.0', '247.0']
Region (nominal, 3 distinct): ['3', '1', '2']
'''

CONTEXT = "Wholesale Customers in Portugal"
TARGET = CuratedTarget(raw_name="Channel", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "Horeca - Hotel / Restaurant / Cafe", "2": "Retail"},)
COLS_TO_DROP = ["V1"]
FEATURES = [CuratedFeature(raw_name="V2", new_name="Annual Spending on Fresh Products"),
            CuratedFeature(raw_name="V3", new_name="Annual Spending on Milk Products"),
            CuratedFeature(raw_name="V4", new_name="Annual Spending on Grocery Products"),
            CuratedFeature(raw_name="V5", new_name="Annual Spending on Frozen Products"),
            CuratedFeature(raw_name="V6", new_name="Annual Spending on Detergents and Paper Products"),
            CuratedFeature(raw_name="V7", new_name="Annual Spending on Delicatessen Products"),
            CuratedFeature(raw_name="Region", value_mapping={"1": "Lisbon", "2": "Porto", "3": "Other Region"}),
           ]