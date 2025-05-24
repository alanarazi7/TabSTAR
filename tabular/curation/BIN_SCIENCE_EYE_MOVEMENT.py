from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: eye_movements
====
Examples: 7608
====
URL: https://www.openml.org/search?type=data&id=44157
====
Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark,  
                          transformed in the same way. This dataset belongs to the "classification on categorical and
                          numerical features" benchmark. Original description: 
 
**Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Jarkko Salojarvi, Kai Puolamaki, Jaana Simola, Lauri Kovanen, Ilpo Kojo, Samuel Kaski. Inferring Relevance from Eye Movements: Feature Extraction. Helsinki University of Technology, Publications in Computer and Information Science, Report A82. 3 March 2005. Data set at http://www.cis.hut.fi/eyechallenge2005/

Competition 1 (preprocessed data)
A straight-forward classification task. We provide pre-computed feature vectors for each word in the eye movement trajectory, with class labels.

The dataset consist of several assignments. Each assignment consists of a question followed by ten sentences (titles of news articles). One of the sentences is the correct answer to the question (C) and five of the sentences are irrelevant to the question (I). Four of the sentences are relevant to the question (R), but they do not answer it.


* Features are in columns, feature vectors in rows.
* Each assignment is a time sequence of 22-dimensional feature vectors.
* The first column is the line number, second the assignment number and the next 22 columns (3 to 24) are the different features. Columns 25 to 27 contain extra information about the example. The training data set contains the classification label in the 28th column: "0" for irrelevant, "1" for relevant and "2" for the correct answer.
* Each example (row) represents a single word. You are asked to return the classification of each read sentence.
* The 22 features provided are commonly used in psychological studies on eye movement. All of them are not necessarily relevant in this context.

The objective of the Challenge is to predict the classification labels (I, R, C).



Please see the technical report for information of eye movements, experimental setup, baseline methods and references:

Jarkko Salojarvi, Kai Puolamaki, Jaana Simola, Lauri Kovanen, Ilpo Kojo, Samuel Kaski. Inferring Relevance from Eye Movements: Feature Extraction. Helsinki University of Technology, Publications in Computer and Information Science, Report A82. 3 March 2005. [PDF]



Modified by TunedIT (converted to ARFF format)


FEATURES

The values in columns marked with an asterisk (*) are same for all occurances of the word.

COL	NAME		DESCRIPTION
1	#line		Line number
2	#assg		Assignment Number
3	fixcount	Number of fixations to the word
4*	firstPassCnt	Number of fixations to the word when it is first encountered
5*	P1stFixation	'1' if fixation occured when the sentence the word was in was encountered the first time
6*	P2stFixation	'1' if fixation occured when the sentence the word was in was encountered the second time
7*	prevFixDur	Duration of previous fixation
8*	firstfixDur	Duration of the first fixation when the word is first encountered
9*	firstPassFixDur	Sum of durations of fixations when the word is first encountered
10*	nextFixDur	Duration of the next fixation when gaze initially moves from the word
11	firstSaccLen	Length of the first saccade
12	lastSaccLen	Distance between fixation on the word and the next fixation
13	prevFixPos	Distance between the first fixation preceding the word and the beginning ot the word
14	landingPos	Distance between the first fixation on the word and the beginning of the word
15	leavingPos	Distance between the last fixation on the word and the beginning of the word
16	totalFixDur	Sum of all durations of fixations to the word
17	meanFixDur	Mean duration of the fixations to the word
18*	nRegressFrom	Number of regressions leaving from the word
19*	regressLen	Sum of durations of regressions initiating from this word
20*	nextWordRegress	'1' if a regression initiated from the following word
21*	regressDur	Sum of durations of the fixations on the word during regression
22	pupilDiamMax	Maximum pupil diameter
23	pupilDiamLag	Maximum pupil diameter 0.5 - 1.5 seconds after the beginning of fixation
24	timePrtctg	First fixation duration divided by the total number of fixations
25	nWordsInTitle	Number of word in the sentence (title) this word is in
26	titleNo		Title number
27	wordNo		Word number (ordinal) in this title
28	label		Classification for training data ('0'=irrelevant, '1'=relevant, '2'=correct)
====
Target Variable: label (nominal, 2 distinct): ['0', '1']
====
Features:

lineNo (numeric, 7608 distinct): ['9684.0', '3630.0', '3670.0', '3662.0', '3661.0', '3659.0', '3657.0', '3656.0', '3654.0', '3645.0']
assgNo (numeric, 331 distinct): ['78.0', '289.0', '136.0', '22.0', '134.0', '293.0', '135.0', '148.0', '44.0', '306.0']
P1stFixation (nominal, 2 distinct): ['1', '0']
P2stFixation (nominal, 2 distinct): ['0', '1']
prevFixDur (numeric, 58 distinct): ['139.0', '99.0', '179.0', '80.0', '119.0', '159.0', '199.0', '219.0', '0.0', '239.0']
firstfixDur (numeric, 59 distinct): ['139.0', '179.0', '99.0', '119.0', '80.0', '159.0', '199.0', '219.0', '239.0', '258.0']
firstPassFixDur (numeric, 94 distinct): ['139.0', '179.0', '99.0', '199.0', '119.0', '80.0', '159.0', '219.0', '239.0', '258.0']
nextFixDur (numeric, 62 distinct): ['139.0', '179.0', '99.0', '119.0', '80.0', '199.0', '159.0', '219.0', '239.0', '258.0']
firstSaccLen (numeric, 6792 distinct): ['0.0', '114.0175', '158.2024', '106.0189', '111.6121', '220.096', '141.8185', '119.877', '171.0117', '66.7308']
lastSaccLen (numeric, 6977 distinct): ['0.0', '220.096', '113.1923', '111.6121', '106.0189', '114.0175', '249.3657', '107.2287', '134.1827', '79.1344']
prevFixPos (numeric, 5911 distinct): ['0.0', '49.0408', '59.0339', '49.0918', '138.4377', '87.0919', '74.3303', '88.0909', '99.7647', '95.2103']
landingPos (numeric, 5390 distinct): ['18.5068', '46.3276', '40.5123', '2.6926', '47.7127', '43.3503', '13.4629', '56.1894', '79.0142', '10.5119']
leavingPos (numeric, 5458 distinct): ['47.5395', '33.2415', '49.0026', '43.1567', '66.3363', '15.2069', '58.1055', '49.3077', '48.5412', '56.5796']
totalFixDur (numeric, 105 distinct): ['139.0', '179.0', '99.0', '199.0', '119.0', '159.0', '219.0', '80.0', '239.0', '258.0']
meanFixDur (numeric, 166 distinct): ['139.0', '179.0', '99.0', '119.0', '159.0', '80.0', '199.0', '219.0', '239.0', '258.0']
regressLen (numeric, 431 distinct): ['0.0', '139.0', '179.0', '99.0', '219.0', '199.0', '119.0', '318.0', '278.0', '358.0']
nextWordRegress (nominal, 2 distinct): ['0', '1']
regressDur (numeric, 249 distinct): ['0.0', '139.0', '179.0', '99.0', '199.0', '119.0', '219.0', '80.0', '278.0', '159.0']
pupilDiamMax (numeric, 3058 distinct): ['0.1916', '0.122', '0.0642', '0.0487', '0.1016', '0.089', '0.1968', '0.0761', '0.0837', '0.0883']
pupilDiamLag (numeric, 2158 distinct): ['0.1708', '0.2636', '0.1033', '0.1572', '0.2085', '0.122', '0.556', '0.0971', '0.1602', '0.1771']
timePrtctg (numeric, 843 distinct): ['0.0133', '0.0139', '0.0143', '0.0119', '0.0179', '0.008', '0.0163', '0.0192', '0.0172', '0.0147']
titleNo (numeric, 10 distinct): ['1', '2', '4', '3', '5', '8', '7', '6', '10', '9']
wordNo (numeric, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
'''

CONTEXT = "Eye Movement Trajectory Classification"
TARGET = CuratedTarget(raw_name="label", new_name="Relevance", task_type=SupervisedTask.BINARY,
                       label_mapping={"0": "Irrelevant", "1": "Relevant"})
COLS_TO_DROP = []
FEATURES = []