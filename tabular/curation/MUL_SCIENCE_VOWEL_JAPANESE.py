from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: JapaneseVowels
====
Examples: 9961
====
URL: https://www.openml.org/search?type=data&id=375
====
Description: **Author**: Mineichi Kudo, Jun Toyama, Masaru Shimbo    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels)    
**Please cite**:   

**Japanese vowels**  
This dataset records 640 time series of 12 LPC cepstrum coefficients taken from nine male speakers.

The data was collected for examining our newly developed classifier for multidimensional curves (multidimensional time series). Nine male speakers uttered two Japanese vowels /ae/ successively. For each utterance, with the analysis parameters described below, we applied 12-degree linear prediction analysis to it to obtain a discrete-time series with 12 LPC cepstrum coefficients. This means that one utterance by a speaker forms a time series whose length is in the range 7-29 and each point of a time series is of 12 features (12 coefficients).

Similar data are available for different utterances /ei/, /iu/, /uo/, /oa/ in addition to /ae/. Please contact the donor if you are interested in using this data.

The number of the time series is 640 in total. We used one set of 270 time series for training and the other set of 370 time series for testing.

Analysis parameters:  
* Sampling rate : 10kHz
* Frame length : 25.6 ms
* Shift length : 6.4ms
* Degree of LPC coefficients : 12

Each line represents 12 LPC coefficients in the increasing order separated by spaces. This corresponds to one analysis
frame. Lines are organized into blocks, which are a set of 7-29 lines separated by blank lines and corresponds to a single speech utterance of /ae/ with 7-29 frames.

Each speaker is a set of consecutive blocks. In ae.train there are 30 blocks for each speaker. Blocks 1-30 represent speaker 1, blocks 31-60 represent speaker 2, and so on up to speaker 9. In ae.test, speakers 1 to 9 have the corresponding number of blocks: 31 35 88 44 29 24 40 50 29. Thus, blocks 1-31 represent speaker 1 (31 utterances of /ae/), blocks 32-66 represent speaker 2 (35 utterances of /ae/), and so on.

**Past Usage**

M. Kudo, J. Toyama and M. Shimbo. (1999). "Multidimensional Curve Classification Using Passing-Through Regions". Pattern Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.

If you publish any work using the dataset, please inform the donor. Use for commercial purposes requires donor permission.

References  

1. http://ips9.main.eng.hokudai.ac.jp/index_e.html
2. mailto:mine@main.eng.hokudai.ac.jp
3. mailto:jun@main.eng.hokudai.ac.jp
4. mailto:shimbo@main.eng.hokudai.ac.jp
5. http://kdd.ics.uci.edu/
6. http://www.ics.uci.edu/
7. http://www.uci.edu/
====
Target Variable: speaker (nominal, 9 distinct): ['3', '4', '7', '1', '8', '2', '6', '9', '5']
====
Features:

utterance (numeric, 88 distinct): ['8', '7', '1', '23', '10', '9', '18', '24', '5', '21']
frame (numeric, 29 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
coefficient1 (numeric, 9935 distinct): ['0.2927', '1.2334', '0.3029', '0.5097', '0.5421', '0.4665', '0.8487', '1.1674', '1.2979', '1.389']
coefficient2 (numeric, 9924 distinct): ['-0.3311', '-0.6532', '-0.6193', '-0.4635', '0.0289', '-0.8224', '-0.4128', '-0.8454', '-0.6705', '-0.6612']
coefficient3 (numeric, 9918 distinct): ['0.269', '-0.1213', '0.4561', '0.2906', '0.4799', '-0.02', '0.4421', '0.4455', '0.3398', '0.1725']
coefficient4 (numeric, 9906 distinct): ['-0.1527', '-0.4325', '-0.2591', '-0.5784', '-0.3029', '-0.7468', '-0.6005', '-0.3345', '-0.4027', '-0.2905']
coefficient5 (numeric, 9922 distinct): ['0.3185', '0.1151', '0.3774', '0.1738', '0.625', '0.3108', '0.3185', '0.1979', '0.1233', '0.6438']
coefficient6 (numeric, 9898 distinct): ['-0.2942', '-0.2616', '-0.0992', '-0.2609', '-0.2293', '-0.3276', '-0.2386', '-0.1468', '-0.0733', '-0.0502']
coefficient7 (numeric, 9876 distinct): ['-0.0293', '-0.0942', '-0.3897', '-0.0342', '-0.2809', '0.0786', '-0.3179', '-0.2221', '-0.17', '-0.1521']
coefficient8 (numeric, 9893 distinct): ['0.1021', '-0.244', '-0.1829', '-0.0176', '-0.0305', '0.0939', '-0.3827', '0.0042', '0.0492', '0.0579']
coefficient9 (numeric, 9892 distinct): ['-0.2663', '-0.2555', '-0.3919', '-0.3367', '-0.1548', '-0.0337', '-0.1294', '-0.1922', '-0.5344', '-0.078']
coefficient10 (numeric, 9857 distinct): ['-0.1692', '-0.1141', '-0.2021', '-0.2639', '-0.0211', '-0.2169', '-0.1316', '-0.1443', '-0.1285', '-0.1928']
coefficient11 (numeric, 9831 distinct): ['0.0448', '-0.1087', '0.0669', '0.0745', '-0.0581', '-0.0579', '0.0077', '-0.1165', '-0.0376', '-0.0351']
coefficient12 (numeric, 9846 distinct): ['0.0633', '0.1185', '0.08', '0.0082', '0.0982', '0.1462', '0.0638', '0.1098', '-0.0239', '0.0833']
'''

CONTEXT = "Japanese Vowels"
TARGET = CuratedTarget(raw_name="speaker", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []