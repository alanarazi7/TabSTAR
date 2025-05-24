from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: BachChoralHarmony
====
Examples: 5665
====
URL: https://www.openml.org/search?type=data&id=4552
====
Description: **Author**: -- Creators: Daniele P. Radicioni and Roberto Esposito  -- Donor: Daniele P. Radicioni (radicion '@' di.unito.it) and Roberto Esposito (esposito '@' di.unito.it)  -- Date: May","2014  
**Source**: UCI  
**Please cite**: D. P. Radicioni and R. Esposito. Advances in Music Information Retrieval, chapter BREVE: an HMPerceptron-Based Chord Recognition System. Studies in Computational Intelligence, Zbigniew W. Ras and Alicja Wieczorkowska (Editors), Springer, 2010.  

Abstract: The data set is composed of 60 chorales (5665 events) by J.S. Bach (1675-1750). Each event of each chorale is labelled using 1 among 101 chord labels and described through 14 features.
Source:

-- Creators: Daniele P. Radicioni and Roberto Esposito 
-- Donor: Daniele P. Radicioni (radicion '@' di.unito.it) and Roberto Esposito (esposito '@' di.unito.it) 
-- Date: May, 2014


Data Set Information:

Pitch classes information has been extracted from MIDI sources downloaded 
from (JSB Chorales)[[Web Link]]. Meter information has 
been computed through the Meter program which is part of the Melisma 
music analyser (Melisma)[[Web Link]]. 
Chord labels have been manually annotated by a human expert.


Attribute Information:

1. Choral ID: corresponding to the file names from (Bach Central)[[Web Link]]. 
2. Event number: index (starting from 1) of the event inside the chorale. 
3-14. Pitch classes: YES/NO depending on whether a given pitch is present. 
Pitch classes/attribute correspondence is as follows: 
C -&gt; 3 
C#/Db -&gt; 4 
D -&gt; 5 
... 
B -&gt; 14 
15. Bass: Pitch class of the bass note 
16. Meter: integers from 1 to 5. Lower numbers denote less accented events, 
higher numbers denote more accented events. 
17. Chord label: Chord resonating during the given event.


Relevant Papers:

1. D. P. Radicioni and R. Esposito. Advances in Music Information Retrieval, 
chapter BREVE: an HMPerceptron-Based Chord Recognition System. Studies 
in Computational Intelligence, Zbigniew W. Ras and Alicja Wieczorkowska 
(Editors), Springer, 2010. 
2. Esposito, R. and Radicioni, D. P., CarpeDiem: Optimizing the Viterbi 
Algorithm and Applications to Supervised Sequential Learning, Journal 
of Machine Learning Research, 10(Aug):1851-1880, 2009.



Citation Request:

D. P. Radicioni and R. Esposito. Advances in Music Information Retrieval, chapter BREVE: an HMPerceptron-Based Chord Recognition System. Studies in Computational Intelligence, Zbigniew W. Ras and Alicja Wieczorkowska (Editors), Springer, 2010.
====
Target Variable: V17 (nominal, 102 distinct): [' D_M', ' G_M', ' C_M', ' F_M', ' A_M', ' BbM', ' E_M', ' A_m', ' E_m', ' B_m']
====
Features:

V1 (nominal, 62 distinct): ['002908ch', '012606bv', '012606b_', '000106b_', '003306b_', '001106b_', '014007b_', '001707b_', '004008b_', '014406b_']
V2 (numeric, 207 distinct): ['1', '22', '24', '25', '27', '28', '29', '30', '31', '32']
V3 (nominal, 2 distinct): [' NO', 'YES']
V4 (nominal, 2 distinct): [' NO', 'YES']
V5 (nominal, 2 distinct): [' NO', 'YES']
V6 (nominal, 2 distinct): [' NO', 'YES']
V7 (nominal, 2 distinct): [' NO', 'YES']
V8 (nominal, 2 distinct): [' NO', 'YES']
V9 (nominal, 2 distinct): [' NO', 'YES']
V10 (nominal, 2 distinct): [' NO', 'YES']
V11 (nominal, 2 distinct): [' NO', 'YES']
V12 (nominal, 2 distinct): [' NO', 'YES']
V13 (nominal, 2 distinct): [' NO', 'YES']
V14 (nominal, 2 distinct): [' NO', 'YES']
V15 (nominal, 16 distinct): ['D', 'A', 'G', 'E', 'C', 'F', 'F#', 'B', 'Bb', 'C#']
V16 (numeric, 5 distinct): ['3', '2', '5', '4', '1']
'''

CONTEXT = "Bach Choral Harmony - Chord Recognition"
TARGET = CuratedTarget(raw_name="V17", new_name="Chord Label", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["V1"]
FEATURES = [CuratedFeature(raw_name="V2", new_name="Event Number"),
            CuratedFeature(raw_name="V3", new_name="Pitch 1 Present"),
            CuratedFeature(raw_name="V4", new_name="Pitch 2 Present"),
            CuratedFeature(raw_name="V5", new_name="Pitch 3 Present"),
            CuratedFeature(raw_name="V6", new_name="Pitch 4 Present"),
            CuratedFeature(raw_name="V7", new_name="Pitch 5 Present"),
            CuratedFeature(raw_name="V8", new_name="Pitch 6 Present"),
            CuratedFeature(raw_name="V9", new_name="Pitch 7 Present"),
            CuratedFeature(raw_name="V10", new_name="Pitch 8 Present"),
            CuratedFeature(raw_name="V11", new_name="Pitch 9 Present"),
            CuratedFeature(raw_name="V12", new_name="Pitch 10 Present"),
            CuratedFeature(raw_name="V13", new_name="Pitch 11 Present"),
            CuratedFeature(raw_name="V14", new_name="Pitch 12 Present"),
            CuratedFeature(raw_name="V15", new_name="Pitch Bass Note"),
            CuratedFeature(raw_name="V16", new_name="Meter - accent level")]