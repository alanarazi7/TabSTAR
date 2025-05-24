from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: pendigits
====
Examples: 10992
====
URL: https://www.openml.org/search?type=data&id=32
====
Description: **Author**: E. Alpaydin, Fevzi. Alimoglu  
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)  
**Please cite**:  [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)

**Pen-Based Recognition of Handwritten Digits**  
We create a digit database by collecting 250 samples from 44 writers. The samples, written by 30 writers, are used for training, cross-validation and writer dependent testing, and the digits written by the other 14 are used for writer independent testing. This database is also available in the UNIPEN format.

We use a WACOM PL-100V pressure sensitive tablet with an integrated LCD display and a cordless stylus. The input and display areas are located in the same place. Attached to the serial port of an Intel 486 based PC, it allows us to collect handwriting samples. The tablet
sends $x$ and $y$ tablet coordinates and pressure level values of the pen at fixed time intervals (sampling rate) of 100 miliseconds. 
 
These writers are asked to write 250 digits in random order inside boxes of 500 by 500 tablet pixel resolution.  Subject are monitored only during the first entry screens. Each screen contains five boxes with the digits to be written displayed above. Subjects are told to write only inside these boxes.  If they make a mistake or are unhappy with their writing, they are instructed to clear the content of a box by using an on-screen button. The first ten digits are ignored because most writers are not familiar with this type of input devices, but subjects are not aware of this. 
 
In our study, we use only ($x, y$) coordinate information. The stylus pressure level values are ignored. First we apply normalization to make our representation invariant to translations and scale distortions. The raw data that we capture from the tablet consist of integer values between 0 and 500 (tablet input box resolution). The new coordinates are such that the coordinate which has the maximum range varies between 0 and 100. Usually $x$ stays in this range, since most characters are taller than they are wide.  

### Attribute information  

In order to train and test our classifiers, we need to represent digits as constant length feature vectors. A commonly used technique leading to good results is resampling the ( x_t, y_t) points. Temporal resampling (points regularly spaced in time) or spatial resampling (points regularly spaced in arc length) can be used here. Raw point data are already regularly spaced in time but the distance between them is variable. Previous research showed that spatial resampling to obtain a constant number of regularly spaced points on the trajectory yields much better performance, because it provides a better alignment between points. Our resampling algorithm uses simple linear interpolation between pairs of points. The resampled digits are represented as a sequence of T points ( x_t, y_t )_{t=1}^T, regularly spaced in arc length, as opposed to the input sequence, which is regularly spaced in time.
 
So, the input vector size is 2*T, two times the number of points resampled. We considered spatial resampling to T=8,12,16 points in our experiments and found that T=8 gave the best trade-off between accuracy and complexity.
 
The way we used the dataset was to use first half of training for actual training, one-fourth for validation and one-fourth for writer-dependent testing. The test set was used for writer-independent testing and is the actual quality measure.
====
Target Variable: class (nominal, 10 distinct): ['2', '4', '0', '1', '7', '6', '3', '5', '8', '9']
====
Features:

input1 (numeric, 101 distinct): ['0.0', '100.0', '32.0', '26.0', '38.0', '19.0', '16.0', '25.0', '23.0', '15.0']
input2 (numeric, 96 distinct): ['100.0', '90.0', '91.0', '88.0', '96.0', '93.0', '85.0', '89.0', '94.0', '82.0']
input3 (numeric, 101 distinct): ['0.0', '100.0', '35.0', '46.0', '39.0', '40.0', '34.0', '38.0', '44.0', '43.0']
input4 (numeric, 98 distinct): ['100.0', '75.0', '78.0', '81.0', '77.0', '80.0', '79.0', '99.0', '82.0', '76.0']
input5 (numeric, 101 distinct): ['0.0', '100.0', '66.0', '68.0', '65.0', '62.0', '60.0', '64.0', '67.0', '61.0']
input6 (numeric, 101 distinct): ['100.0', '0.0', '78.0', '75.0', '79.0', '72.0', '76.0', '77.0', '74.0', '73.0']
input7 (numeric, 101 distinct): ['0.0', '100.0', '68.0', '56.0', '61.0', '70.0', '53.0', '57.0', '67.0', '62.0']
input8 (numeric, 101 distinct): ['0.0', '100.0', '35.0', '34.0', '40.0', '39.0', '43.0', '36.0', '38.0', '33.0']
input9 (numeric, 101 distinct): ['100.0', '0.0', '84.0', '88.0', '94.0', '79.0', '71.0', '60.0', '41.0', '82.0']
input10 (numeric, 101 distinct): ['0.0', '75.0', '35.0', '44.0', '36.0', '45.0', '34.0', '43.0', '42.0', '41.0']
input11 (numeric, 101 distinct): ['100.0', '0.0', '88.0', '91.0', '87.0', '86.0', '92.0', '82.0', '78.0', '90.0']
input12 (numeric, 101 distinct): ['0.0', '50.0', '100.0', '16.0', '15.0', '9.0', '18.0', '12.0', '17.0', '20.0']
input13 (numeric, 101 distinct): ['100.0', '50.0', '51.0', '49.0', '53.0', '52.0', '48.0', '55.0', '46.0', '56.0']
input14 (numeric, 101 distinct): ['0.0', '100.0', '25.0', '1.0', '32.0', '2.0', '24.0', '34.0', '27.0', '4.0']
input15 (numeric, 101 distinct): ['100.0', '0.0', '7.0', '8.0', '9.0', '5.0', '6.0', '16.0', '10.0', '19.0']
input16 (numeric, 101 distinct): ['0.0', '100.0', '1.0', '2.0', '5.0', '3.0', '4.0', '7.0', '6.0', '8.0']
'''

CONTEXT = "Anonymized Dataset - Pendigits - Handwritten Digits"
TARGET = CuratedTarget(raw_name="class", new_name="Digit", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []