from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: collins
====
Examples: 1000
====
URL: https://www.openml.org/search?type=data&id=40971
====
Description: **Author**: Jeff Collins  
**Source**: [StatLib](http://lib.stat.cmu.edu/datasets/collins.txt)  
**Please cite**: None  

Data used in an analysis of the Brown and Frown corpora for my doctoral dissertation titled ``Variations in Written English: Characterizing Authors' Rhetorical Language Choices Across Corpora of Published Texts" (Completed at Carnegie Mellon Univ, 2003).  The source of the corpora was the ICAME CD-ROM  (get info at <http>).

The data were generated from the texts using tagging and visualization software, Docuscope.

The first row is the variable names. The genre of each text (assigned by the Brown corpus compilers) is in 'Genre' column and the corpus is listed in the 'corpus' column with 1=Brown and 2=Frown corpus.

The dataset may be freely used and distributed for non-commercial purposes.

Note: The Genre and Corpus values together make up the target, and the Countr just counts documents within each counter, so they should probably be ignored.
====
Target Variable: Corp.Genre (nominal, 30 distinct): ['109', '209', '207', '107', '206', '106', '101', '201', '205', '105']
====
Features:

FirstPerson (numeric, 228 distinct): ['0.0', '0.04', '0.09', '0.13', '0.17', '0.22', '0.08', '0.26', '0.3', '0.38']
InnerThinking (numeric, 326 distinct): ['2.39', '3.0', '2.42', '2.62', '2.58', '2.45', '2.26', '2.66', '3.09', '2.72']
ThinkPositive (numeric, 177 distinct): ['0.3', '0.62', '0.74', '0.57', '0.79', '0.67', '0.44', '0.58', '0.43', '0.51']
ThinkNegative (numeric, 269 distinct): ['1.21', '0.72', '1.28', '1.46', '1.08', '0.99', '1.61', '1.01', '0.58', '0.93']
ThinkAhead (numeric, 205 distinct): ['0.93', '0.97', '0.84', '1.01', '1.36', '1.23', '1.18', '0.96', '0.81', '1.48']
ThinkBack (numeric, 143 distinct): ['0.22', '0.13', '0.26', '0.38', '0.49', '0.35', '0.43', '0.48', '0.4', '0.25']
Reasoning (numeric, 312 distinct): ['2.73', '2.75', '2.07', '2.63', '3.18', '2.27', '2.36', '2.51', '2.41', '1.82']
Share_SocTies (numeric, 339 distinct): ['2.24', '2.14', '1.81', '1.59', '1.57', '2.11', '1.76', '1.44', '1.2', '1.37']
Direct_Activity (numeric, 98 distinct): ['0.13', '0.17', '0.09', '0.04', '0.22', '0.26', '0.21', '0.3', '0.08', '0.0']
Interacting (numeric, 220 distinct): ['0.0', '0.04', '0.13', '0.09', '0.17', '0.26', '0.22', '0.43', '0.18', '0.08']
Notifying (numeric, 273 distinct): ['2.72', '2.28', '2.61', '3.04', '2.45', '2.4', '1.87', '2.78', '2.15', '3.19']
LinearGuidance (numeric, 562 distinct): ['3.9', '5.91', '3.43', '3.46', '3.75', '3.66', '4.43', '4.09', '3.56', '2.94']
WordPicture (numeric, 577 distinct): ['6.3', '5.85', '3.72', '4.85', '4.34', '4.56', '6.58', '4.57', '6.39', '3.85']
SpaceInterval (numeric, 275 distinct): ['0.93', '0.87', '0.94', '0.69', '0.6', '1.08', '0.83', '0.86', '1.11', '0.88']
Motion (numeric, 148 distinct): ['0.09', '0.04', '0.17', '0.22', '0.26', '0.13', '0.0', '0.18', '0.21', '0.35']
PastEvents (numeric, 350 distinct): ['1.13', '1.4', '1.31', '0.99', '1.27', '1.33', '1.03', '2.2', '0.87', '1.54']
TimeInterval (numeric, 216 distinct): ['1.31', '0.99', '1.02', '1.26', '0.92', '1.0', '1.14', '1.47', '1.25', '1.33']
ShiftingEvents (numeric, 151 distinct): ['0.92', '0.44', '0.77', '0.87', '0.4', '0.52', '0.73', '0.6', '0.81', '0.89']
Text_Coverage (numeric, 793 distinct): ['29.91', '32.35', '26.74', '28.84', '29.12', '30.39', '31.16', '26.0', '29.32', '30.82']
'''

CONTEXT = "Brown Frown Corpora Collins"
TARGET = CuratedTarget(raw_name="Corp.Genre", new_name="Corpus Genre", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []