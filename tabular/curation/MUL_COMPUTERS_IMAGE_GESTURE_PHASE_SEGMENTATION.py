from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: GesturePhaseSegmentationProcessed
====
Examples: 9873
====
URL: https://www.openml.org/search?type=data&id=4538
====
Description: **Author**: Renata Cristina Barros Madeo (Madeo","R. C. B.)  Priscilla Koch Wagner (Wagner","P. K.)  Sarajane Marques Peres (Peres","S. M.)  {renata.si","priscilla.wagner","sarajane} at usp.br  http://each.uspnet.usp.br/sarajane/  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/gesture+phase+segmentation)  
**Please cite**: Please refer to the [Machine Learning Repository's citation policy](https://archive.ics.uci.edu/ml/citation_policy.html). Additionally, the authors require a citation to one or more publications from those cited as relevant papers.  

Creators: 
Renata Cristina Barros Madeo (Madeo, R. C. B.) 
Priscilla Koch Wagner (Wagner, P. K.) 
Sarajane Marques Peres (Peres, S. M.) 
{renata.si, priscilla.wagner, sarajane} at usp.br 
http://each.uspnet.usp.br/sarajane/ 

Donor: 
University of Sao Paulo - Brazil

Data Set Information:

The dataset is composed by features extracted from 7 videos with people gesticulating, aiming at studying Gesture Phase Segmentation. 
Each video is represented by two files: a raw file, which contains the position of hands, wrists, head and spine of the user in each frame; and a processed file, which contains velocity and acceleration of hands and wrists. See the data set description for more information on the dataset.

Attribute Information:

Raw files: 18 numeric attributes (double), a timestamp and a class attribute (nominal). 
Processed files: 32 numeric attributes (double) and a class attribute (nominal). 
A feature vector with up to 50 numeric attributes can be generated with the two files mentioned above.

This is the processed data set with the following feature description:

   Processed files:

   1. Vectorial velocity of left hand (x coordinate)
   2. Vectorial velocity of left hand (y coordinate)
   3. Vectorial velocity of left hand (z coordinate)
   4. Vectorial velocity of right hand (x coordinate)
   5. Vectorial velocity of right hand (y coordinate)
   6. Vectorial velocity of right hand (z coordinate)
   7. Vectorial velocity of left wrist (x coordinate)
   8. Vectorial velocity of left wrist (y coordinate)
   9. Vectorial velocity of left wrist (z coordinate)
   10. Vectorial velocity of right wrist (x coordinate)
   11. Vectorial velocity of right wrist (y coordinate)
   12. Vectorial velocity of right wrist (z coordinate)
   13. Vectorial acceleration of left hand (x coordinate)
   14. Vectorial acceleration of left hand (y coordinate)
   15. Vectorial acceleration of left hand (z coordinate)
   16. Vectorial acceleration of right hand (x coordinate)
   17. Vectorial acceleration of right hand (y coordinate)
   18. Vectorial acceleration of right hand (z coordinate)
   19. Vectorial acceleration of left wrist (x coordinate)
   20. Vectorial acceleration of left wrist (y coordinate)
   21. Vectorial acceleration of left wrist (z coordinate)
   22. Vectorial acceleration of right wrist (x coordinate)
   23. Vectorial acceleration of right wrist (y coordinate)
   24. Vectorial acceleration of right wrist (z coordinate)
   25. Scalar velocity of left hand
   26. Scalar velocity of right hand
   27. Scalar velocity of left wrist
   28. Scalar velocity of right wrist
   29. Scalar velocity of left hand
   30. Scalar velocity of right hand
   31. Scalar velocity of left wrist
   32. Scalar velocity of right wrist
   33. phase:
       - D (rest position, from portuguese "descanso")
       - P (preparation)
       - S (stroke)
       - H (hold)
       - R (retraction)

Relevant Papers:

1. Madeo, R. C. B. ; Lima, C. A. M. ; PERES, S. M. . Gesture Unit Segmentation using Support Vector Machines: Segmenting 
Gestures from Rest Positions. In: Symposium on Applied Computing (SAC), 2013, Coimbra. Proceedings of the 28th Annual 
ACM Symposium on Applied Computing (SAC), 2013. p. 46-52. 
* In this paper, the videos A1 and A2 were studied. 

2. Wagner, P. K. ; PERES, S. M. ; Madeo, R. C. B. ; Lima, C. A. M. ; Freitas, F. A. . Gesture Unit Segmentation Using 
Spatial-Temporal Information and Machine Learning. In: 27th Florida Artificial Intelligence Research Society Conference 
(FLAIRS), 2014, Pensacola Beach. Proceedings of the 27th Florida Artificial Intelligence Research Society Conference 
(FLAIRS). Palo Alto : The AAAI Press, 2014. p. 101-106. 
* In this paper, the videos A1, A2, A3, B1, B3, C1 and C3 were studied. 

3. Madeo, R. C. B.. Support Vector Machines and Gesture Analysis: incorporating temporal aspects (in Portuguese). Master 
Thesis - Universidade de Sao Paulo, Sao Paulo Researcher Foundation. 2013. 
* In this document, the videos named B1 and B3 in the document correspond to videos C1 and C3 in this dataset. Only 
five videos were explored in this document: A1, A2, A3, C1 and C3. 

4. Wagner, P. K. ; Madeo, R. C. B. ; PERES, S. M. ; Lima, C. A. M. . Segmenta&Atilde;&sect;ao de Unidades Gestuais com Multilayer 
Perceptrons (in Portuguese). In: Encontro Nacional de Inteligencia Artificial e Computacional (ENIAC), 2013, Fortaleza. 
Anais do X Encontro Nacional de Inteligencia Artificial e Computacional (ENIAC), 2013. 
* In this paper, the videos A1, A2 and A3 were studied.



Citation Request:

Please refer to the Machine Learning Repository's citation policy. 
Additionally, the authors require a citation to one or more publications from those cited as relevant papers.
====
Target Variable: Phase (nominal, 5 distinct): ['S', 'D', 'P', 'R', 'H']
====
Features:

X1 (numeric, 9822 distinct): ['-0.0002', '-0.0001', '-0.0002', '0.0008', '-0.0005', '-0.0', '0.0008', '0.0005', '-0.0004', '0.0007']
X2 (numeric, 9826 distinct): ['-0.0', '-0.0003', '0.0002', '-0.0007', '-0.0008', '-0.0', '0.0001', '0.0001', '0.0025', '-0.0062']
X3 (numeric, 9625 distinct): ['0.0', '0.0001', '0.0', '-0.0001', '-0.0', '0.0', '0.0', '-0.0001', '0.0001', '-0.0001']
X4 (numeric, 9810 distinct): ['-0.0073', '0.002', '0.0004', '0.0007', '-0.0017', '0.0', '-0.0002', '0.0003', '-0.0006', '0.0003']
X5 (numeric, 9840 distinct): ['-0.0003', '0.0016', '-0.0011', '0.0001', '-0.0009', '0.0033', '0.0012', '-0.0', '-0.0066', '-0.0018']
X6 (numeric, 9703 distinct): ['-0.0003', '-0.0', '-0.0', '-0.0', '0.0', '0.0002', '0.0', '0.0001', '-0.0', '-0.0']
X7 (numeric, 9816 distinct): ['0.0014', '0.0014', '0.0001', '0.002', '0.0019', '-0.0', '-0.0', '0.0001', '-0.0', '-0.001']
X8 (numeric, 9810 distinct): ['0.0004', '0.0002', '0.0', '0.0002', '-0.0003', '-0.0001', '-0.0', '-0.0002', '0.0003', '0.0003']
X9 (numeric, 9595 distinct): ['0.0001', '-0.0', '0.0001', '-0.0001', '-0.0', '-0.0001', '-0.0001', '0.0001', '-0.0', '-0.0001']
X10 (numeric, 9817 distinct): ['-0.0006', '-0.0001', '-0.0', '-0.0004', '0.0021', '-0.0001', '-0.0002', '-0.0001', '0.0002', '0.0006']
X11 (numeric, 9827 distinct): ['0.0075', '0.0001', '0.0002', '-0.0031', '-0.0007', '0.0', '0.0004', '0.0001', '-0.0005', '-0.0001']
X12 (numeric, 9664 distinct): ['0.0', '-0.0001', '0.0', '0.0001', '0.0', '-0.0', '0.0001', '-0.0', '-0.0', '-0.0']
X13 (numeric, 9466 distinct): ['-0.0', '0.0', '0.0', '0.0', '-0.0002', '-0.0', '-0.0', '0.0', '-0.0', '-0.0']
X14 (numeric, 9489 distinct): ['-0.0', '0.0', '-0.0', '0.0', '-0.0', '-0.0001', '0.0', '0.0', '0.0', '-0.0']
X15 (numeric, 8395 distinct): ['0.0', '-0.0', '-0.0', '0.0', '0.0', '-0.0', '0.0', '0.0', '0.0', '-0.0']
X16 (numeric, 9557 distinct): ['-0.0', '0.0001', '0.0', '-0.0', '0.0', '-0.0', '-0.0', '-0.0001', '-0.0', '0.0']
X17 (numeric, 9584 distinct): ['-0.0', '0.0', '-0.0', '0.0001', '-0.0', '-0.0', '-0.0', '0.0002', '0.0001', '0.0']
X18 (numeric, 8724 distinct): ['-0.0', '-0.0', '-0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0', '0.0', '-0.0']
X19 (numeric, 9425 distinct): ['-0.0', '-0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0', '0.0001', '-0.0', '0.0']
X20 (numeric, 9380 distinct): ['0.0', '0.0', '-0.0', '-0.0', '0.0001', '0.0', '-0.0', '0.0', '-0.0', '0.0']
X21 (numeric, 8304 distinct): ['0.0', '-0.0', '-0.0', '-0.0', '0.0', '-0.0', '0.0', '-0.0', '-0.0', '-0.0']
X22 (numeric, 9473 distinct): ['-0.0', '0.0', '0.0001', '0.0001', '0.0', '-0.0', '0.0', '0.0', '0.0', '-0.0']
X23 (numeric, 9533 distinct): ['-0.0', '-0.0', '0.0', '0.0', '0.0', '-0.0', '0.0', '-0.0', '-0.0002', '0.0']
X24 (numeric, 8714 distinct): ['-0.0', '-0.0', '0.0', '-0.0', '-0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0']
X25 (numeric, 9831 distinct): ['0.0003', '0.0003', '0.0002', '0.0063', '0.0009', '0.0003', '0.0007', '0.0007', '0.0006', '0.0003']
X26 (numeric, 9844 distinct): ['0.0057', '0.0002', '0.0026', '0.0079', '0.0006', '0.0138', '0.0007', '0.0003', '0.0015', '0.0011']
X27 (numeric, 9824 distinct): ['0.0005', '0.0009', '0.0005', '0.0', '0.0007', '0.0004', '0.0033', '0.0011', '0.0003', '0.0003']
X28 (numeric, 9827 distinct): ['0.0115', '0.0097', '0.0007', '0.0058', '0.0147', '0.0005', '0.0029', '0.0004', '0.0004', '0.0036']
X29 (numeric, 9509 distinct): ['0.0', '0.0', '0.0', '0.0001', '0.0', '0.0', '0.0', '0.0001', '0.0', '0.0']
X30 (numeric, 9635 distinct): ['0.0', '0.0001', '0.0', '0.0001', '0.0', '0.0001', '0.0003', '0.0014', '0.0001', '0.0']
X31 (numeric, 9493 distinct): ['0.0001', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0001', '0.0001']
X32 (numeric, 9576 distinct): ['0.0', '0.0', '0.0', '0.0001', '0.0', '0.0', '0.0001', '0.0', '0.0004', '0.0007']
'''

CONTEXT = "Gesture Phase Segmentation from Videos"
COLS_TO_DROP = []
TARGET = CuratedTarget(raw_name="Phase", new_name="Gesture Phase", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'S': 'Stroke', 'D': 'Rest Position', 'P': 'Preparation', 'R': 'Retraction', 'H': 'Hold'})
FEATURES = [
            CuratedFeature(raw_name="X1", new_name="Vectorial velocity of left hand (x coordinate)"),
            CuratedFeature(raw_name="X2", new_name="Vectorial velocity of left hand (y coordinate)"),
            CuratedFeature(raw_name="X3", new_name="Vectorial velocity of left hand (z coordinate)"),
            CuratedFeature(raw_name="X4", new_name="Vectorial velocity of right hand (x coordinate)"),
            CuratedFeature(raw_name="X5", new_name="Vectorial velocity of right hand (y coordinate)"),
            CuratedFeature(raw_name="X6", new_name="Vectorial velocity of right hand (z coordinate)"),
            CuratedFeature(raw_name="X7", new_name="Vectorial velocity of left wrist (x coordinate)"),
            CuratedFeature(raw_name="X8", new_name="Vectorial velocity of left wrist (y coordinate)"),
            CuratedFeature(raw_name="X9", new_name="Vectorial velocity of left wrist (z coordinate)"),
            CuratedFeature(raw_name="X10", new_name="Vectorial velocity of right wrist (x coordinate)"),
            CuratedFeature(raw_name="X11", new_name="Vectorial velocity of right wrist (y coordinate)"),
            CuratedFeature(raw_name="X12", new_name="Vectorial velocity of right wrist (z coordinate)"),
            CuratedFeature(raw_name="X13", new_name="Vectorial acceleration of left hand (x coordinate)"),
            CuratedFeature(raw_name="X14", new_name="Vectorial acceleration of left hand (y coordinate)"),
            CuratedFeature(raw_name="X15", new_name="Vectorial acceleration of left hand (z coordinate)"),
            CuratedFeature(raw_name="X16", new_name="Vectorial acceleration of right hand (x coordinate)"),
            CuratedFeature(raw_name="X17", new_name="Vectorial acceleration of right hand (y coordinate)"),
            CuratedFeature(raw_name="X18", new_name="Vectorial acceleration of right hand (z coordinate)"),
            CuratedFeature(raw_name="X19", new_name="Vectorial acceleration of left wrist (x coordinate)"),
            CuratedFeature(raw_name="X20", new_name="Vectorial acceleration of left wrist (y coordinate)"),
            CuratedFeature(raw_name="X21", new_name="Vectorial acceleration of left wrist (z coordinate)"),
            CuratedFeature(raw_name="X22", new_name="Vectorial acceleration of right wrist (x coordinate)"),
            CuratedFeature(raw_name="X23", new_name="Vectorial acceleration of right wrist (y coordinate)"),
            CuratedFeature(raw_name="X24", new_name="Vectorial acceleration of right wrist (z coordinate)"),
            CuratedFeature(raw_name="X25", new_name="Scalar velocity of left hand 1"),
            CuratedFeature(raw_name="X26", new_name="Scalar velocity of right hand 1"),
            CuratedFeature(raw_name="X27", new_name="Scalar velocity of left wrist 1"),
            CuratedFeature(raw_name="X28", new_name="Scalar velocity of right wrist 1"),
            CuratedFeature(raw_name="X29", new_name="Scalar velocity of left hand 2"),
            CuratedFeature(raw_name="X30", new_name="Scalar velocity of right hand 2"),
            CuratedFeature(raw_name="X31", new_name="Scalar velocity of left wrist 2"),
            CuratedFeature(raw_name="X32", new_name="Scalar velocity of right wrist 2")
            ]

