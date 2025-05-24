from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: ldpa
====
Examples: 164860
====
URL: https://www.openml.org/search?type=data&id=1483
====
Description: **Author**:   

**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity)

**Please cite**:  B. Kaluza, V. Mirchevska, E. Dovgan, M. Lustrek, M. Gams, An Agent-based Approach to Care in Independent Living, International Joint Conference on Ambient Intelligence (AmI-10), Malaga, Spain, In press

Dataset Title: 

Localization Data for Person Activity Data Set

Abstract: 

Data contains recordings of five people performing different activities. Each person wore four sensors (tags) while performing the same scenario five times.

Source:

- Creators: Mitja Lustrek (mitja.lustrek '@' ijs.si), Bostjan Kaluza (bostjan.kaluza '@' ijs.si), Rok Piltaver (rok.piltaver '@' ijs.si), Jana Krivec (jana.krivec '@' ijs.si), Vedrana Vidulin (vedrana.vidulin '@' ijs.si) 
- Jozef Stefan Institute, Jamova cesta 39, 1000 Ljubljana, Slovenija 
- Donor: Bozidara Cvetkovic (boza.cvetkovic '@' ijs.si) 
- Jozef Stefan Institute, Jamova cesta 39, 1000 Ljubljana, Slovenija 
- Date received: October, 2010

Data Set Information:

People used for recording of the data were wearing four tags (ankle left, ankle right, belt and chest). 
Each instance is a localization data for one of the tags. The tag can be identified by one of the attributes.


Attribute Information:

Instance example: A01,020-000-033-111,633790226057226795,27.05.2009 14:03:25:723,4.292500972747803,2.0738532543182373,1.36650812625885,walking 

1) Sequence Name {A01,A02,A03,A04,A05,B01,B02,B03,B04,B05,C01,C02,C03,C04,C05,D01,D02,D03,D04,D05,E01,E02,E03,E04,E05} (Nominal) 
- A, B, C, D, E = 5 people 
2) Tag identificator {010-000-024-033,020-000-033-111,020-000-032-221,010-000-030-096} (Nominal) 
- ANKLE_LEFT = 010-000-024-033 
- ANKLE_RIGHT = 010-000-030-096 
- CHEST = 020-000-033-111 
- BELT = 020-000-032-221 
3) timestamp (Numeric) all unique 
4) date FORMAT = dd.MM.yyyy HH:mm:ss:SSS (Date) 
5) x coordinate of the tag (Numeric) 
6) y coordinate of the tag (Numeric) 
7) z coordinate of the tag (Numeric) 
8) activity {walking,falling,'lying down',lying,'sitting down',sitting,'standing up from lying','on all fours','sitting on the ground','standing up from sitting','standing up from sitting on the ground'} (Nominal) 

Relevant Papers:

B. Kaluza, V. Mirchevska, E. Dovgan, M. Lustrek, M. Gams, An Agent-based Approach to Care in Independent Living, International Joint Conference on Ambient Intelligence (AmI-10), Malaga, Spain, In press
====
Target Variable: Class (nominal, 11 distinct): ['4', '3', '7', '10', '9', '5', '6', '1', '2', '8']
====
Features:

V1 (nominal, 5 distinct): ['5', '4', '3', '2', '1']
V2 (nominal, 4 distinct): ['1', '3', '2', '4']
V3 (numeric, 164859 distinct): ['30493.0', '105794.0', '64385.0', '64377.0', '64378.0', '64379.0', '64380.0', '64381.0', '64382.0', '64383.0']
V4 (numeric, 164834 distinct): ['55898.0', '30489.0', '152064.0', '125742.0', '62474.0', '92151.0', '7113.0', '107378.0', '14336.0', '19251.0']
V5 (numeric, 163802 distinct): ['108168.0', '120951.0', '98996.0', '73138.0', '118140.0', '102678.0', '86238.0', '53273.0', '110830.0', '155913.0']
V6 (numeric, 163689 distinct): ['105109.0', '64526.0', '148459.0', '48249.0', '155590.0', '22602.0', '96415.0', '126575.0', '67875.0', '118260.0']
V7 (numeric, 164482 distinct): ['105296.0', '123786.0', '67751.0', '82564.0', '119023.0', '90608.0', '118014.0', '119771.0', '105086.0', '72016.0']
'''

CONTEXT = "Activity Identification from Localization Data"
TARGET = CuratedTarget(raw_name="Class", new_name="Activity", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'walking',
                                      '2': 'falling',
                                      '3': 'lying down',
                                      '4': 'lying',
                                      '5': 'sitting down',
                                      '6': 'sitting',
                                      '7': 'standing up from lying',
                                      '8': 'on all fours',
                                      '9': 'sitting on the ground',
                                      '10': 'standing up from sitting',
                                      '11': 'standing up from sitting on the ground'
                                      })
COLS_TO_DROP = ['V4']
FEATURES = [CuratedFeature(raw_name='V1', new_name='Person'),
            CuratedFeature(raw_name='V2', new_name='Tag',
                           value_mapping={'1': 'Left Ankle', '2': 'Right Ankle', '3': 'Chest', '4': 'Belt'}),
            CuratedFeature(raw_name='V3', new_name='Timestamp', feat_type=FeatureType.DATE),
            CuratedFeature(raw_name='V5', new_name='X Coordinate'),
            CuratedFeature(raw_name='V6', new_name='Y Coordinate'),
            CuratedFeature(raw_name='V7', new_name='Z Coordinate'),
            ]