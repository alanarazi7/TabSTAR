from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: vowel
====
Examples: 990
====
URL: https://www.openml.org/search?type=data&id=307
====
Description: **Author**: Peter Turney (peter@ai.iit.nrc.ca)   
**Source**: [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/) - date unknown  
**Please cite**: [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)

**Vowel Recognition (Deterding data)**
Speaker independent recognition of the eleven steady state vowels of British English using a specified training set of lpc derived log area ratios.
Collected by David Deterding (data and non-connectionist analysis), Mahesan Niranjan (first connectionist analysis), Tony Robinson (description, program, data, and results)

A very comprehensive description including comments by the authors can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel.names)

The problem is specified by the accompanying data file, "vowel.data".  This
consists of a three dimensional array: voweldata [speaker, vowel, input].
The speakers are indexed by integers 0-89.  (Actually, there are fifteen
individual speakers, each saying each vowel six times.)  The vowels are
indexed by integers 0-10.  For each utterance, there are ten floating-point
input values, with array indices 0-9.

The problem is to train the network as well as possible using only on data
from "speakers" 0-47, and then to test the network on speakers 48-89,
reporting the number of correct classifications in the test set.

For a more detailed explanation of the problem, see the excerpt from Tony
Robinson's Ph.D. thesis in the COMMENTS section.  In Robinson's opinion,
connectionist problems fall into two classes, the possible and the
impossible.  He is interested in the latter, by which he means problems
that have no exact solution.  Thus the problem here is not to see how fast
a network can be trained (although this is important), but to maximise a
less than perfect performance.

#### METHODOLOGY

Report the number of test vowels classified correctly, (i.e. the number of
occurences when distance of the correct output to the actual output was the
smallest of the set of distances from the actual output to all possible
target outputs).

Though this is not the focus of Robinson's study, it would also be useful
to report how long the training took (measured in pattern presentations or
with a rough count of floating-point operations required) and what level of
success was achieved on the training and testing data after various amounts
of training.  Of course, the network topology and algorithm used should be
precisely described as well.

#### VARIATIONS

This benchmark is proposed to encourage the exploration of different node
types.  Please theorise/experiment/hack.  The author (Robinson) will try to
correspond by email if requested.  In particular there has been some
discussion recently on the use of a cross-entropy distance measure, and it
would be interesting to see results for that.

#### Notes

1. Each of these numbers is based on a single trial with random starting
weights.  More trials would of course be preferable, but the computational
facilities available to Robinson were limited.

2. Graphs are given in Robinson's thesis showing test-set performance vs.
epoch count for some of the training runs.  In most cases, performance
peaks at around 250 correct, after which performance decays to different
degrees.  The numbers given above are final performance figures after about
3000 trials, not the peak performance obtained during the run.

#### REFERENCES

[Deterding89] D. H. Deterding, 1989, University of Cambridge, "Speaker
 Normalisation for Automatic Speech Recognition", submitted for PhD.

[NiranjanFallside88] M. Niranjan and F. Fallside, 1988, Cambridge University
 Engineering Department, "Neural Networks and Radial Basis Functions in
 Classifying Static Speech Patterns", CUED/F-INFENG/TR.22.

[RenalsRohwer89-ijcnn] Steve Renals and Richard Rohwer, "Phoneme
 Classification Experiments Using Radial Basis Functions", Submitted to
 the International Joint Conference on Neural Networks, Washington,
 1989.

[RabinerSchafer78] L. R. Rabiner and R. W. Schafer, Englewood Cliffs, New
 Jersey, 1978, Prentice Hall, "Digital Processing of Speech Signals".

[PragerFallside88] R. W. Prager and F. Fallside, 1988, Cambridge University
 Engineering Department, "The Modified Kanerva Model for Automatic
 Speech Recognition", CUED/F-INFENG/TR.6.

[BroomheadLowe88] D. Broomhead and D. Lowe, 1988, Royal Signals and Radar
 Establishment, Malvern, "Multi-variable Interpolation and Adaptive
 Networks", RSRE memo, #4148.

[RobinsonNiranjanFallside88-tr] A. J. Robinson and M. Niranjan and F. 
   Fallside, 1988, Cambridge University Engineering Department,
 "Generalising the Nodes of the Error Propagation Network",
 CUED/F-INFENG/TR.25.

[Robinson89] A. J. Robinson, 1989, Cambridge University Engineering
 Department, "Dynamic Error Propagation Networks".

[McCullochAinsworth88] N. McCulloch and W. A. Ainsworth, Proceedings of
 Speech'88, Edinburgh, 1988, "Speaker Independent Vowel Recognition
 using a Multi-Layer Perceptron".

[RobinsonFallside88-neuro] A. J. Robinson and F. Fallside, 1988, Proceedings
 of nEuro'88, Paris, June, "A Dynamic Connectionist Model for Phoneme
 Recognition.


#### Notes
* This is version 2. Version 1 is hidden because it includes a feature dividing the data in train and test set. In OpenML this information is explicitly available in the corresponding task.
====
Target Variable: Class (nominal, 11 distinct): ['hid', 'hId', 'hEd', 'hAd', 'hYd', 'had', 'hOd', 'hod', 'hUd', 'hud']
====
Features:

Speaker_Number (nominal, 15 distinct): ['Andrew', 'Bill', 'David', 'Mark', 'Jo', 'Kate', 'Penny', 'Rose', 'Mike', 'Nick']
Sex (nominal, 2 distinct): ['Male', 'Female']
Feature_0 (numeric, 853 distinct): ['-3.242', '-2.973', '-4.316', '-4.052', '-3.065', '-3.661', '-4.039', '-3.034', '-4.047', '-4.471']
Feature_1 (numeric, 877 distinct): ['3.582', '2.141', '1.01', '2.091', '1.952', '1.724', '1.784', '0.777', '1.925', '1.6']
Feature_2 (numeric, 815 distinct): ['-0.389', '0.157', '-0.716', '-0.816', '-1.593', '-1.029', '-0.433', '-0.899', '-1.142', '-1.13']
Feature_3 (numeric, 836 distinct): ['0.234', '0.147', '0.699', '-0.396', '0.814', '0.555', '0.214', '-0.343', '-0.364', '-0.152']
Feature_4 (numeric, 803 distinct): ['0.044', '-0.131', '0.126', '0.688', '-0.194', '-0.298', '-0.316', '-0.372', '0.234', '-0.484']
Feature_5 (numeric, 798 distinct): ['1.047', '0.941', '0.381', '1.045', '0.78', '1.125', '0.498', '0.159', '0.081', '0.329']
Feature_6 (numeric, 748 distinct): ['0.12', '-0.317', '0.162', '0.067', '0.049', '0.187', '0.197', '-0.328', '-0.061', '0.144']
Feature_7 (numeric, 794 distinct): ['-0.175', '0.502', '0.036', '-0.468', '-0.282', '-0.137', '0.172', '-0.231', '0.676', '0.613']
Feature_8 (numeric, 788 distinct): ['0.023', '-0.358', '-0.283', '-0.677', '0.387', '-0.776', '-0.162', '-0.54', '-0.029', '-0.114']
Feature_9 (numeric, 775 distinct): ['-0.301', '0.41', '-0.456', '0.168', '-0.267', '-0.343', '-0.548', '-0.322', '-0.118', '-0.25']
'''

CONTEXT = "Vowel Speech Recognition"
TARGET = CuratedTarget(raw_name="Class", new_name="Vowel", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []