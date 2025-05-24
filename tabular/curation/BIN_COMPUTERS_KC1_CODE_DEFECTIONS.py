from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: kc1
====
Examples: 2109
====
URL: https://www.openml.org/search?type=data&id=1067
====
Description: **Author**: Mike Chapman, NASA  
**Source**: [tera-PROMISE](http://openscience.us/repo/defect/mccabehalsted/kc1.html) - 2004  
**Please cite**: Sayyad Shirabad, J. and Menzies, T.J. (2005) The PROMISE Repository of Software Engineering Databases. School of Information Technology and Engineering, University of Ottawa, Canada.  
  
**KC1 Software defect prediction**  
One of the NASA Metrics Data Program defect data sets. Data from software for storage management for receiving and processing ground data. Data comes from McCabe and features extractors of source code.  These features were defined in the 70s in an attempt to objectively characterize code features that are associated with software quality.

### Attribute Information  

1. loc             : numeric % McCabe's line count of code
2. v(g)            : numeric % McCabe "cyclomatic complexity"
3. ev(g)           : numeric % McCabe "essential complexity"
4. iv(g)           : numeric % McCabe "design complexity"
5. n               : numeric % total operators + operands
6. v               : numeric % "volume"
7. l               : numeric % "program length"
8. d               : numeric % "difficulty"
9. i               : numeric % "intelligence"
10. e               : numeric % "effort"
11. b               : numeric % 
12. t               : numeric % Halstead's time estimator
13. lOCode          : numeric % Halstead's line count
14. lOComment       : numeric % Halstead's count of lines of comments
15. lOBlank         : numeric % Halstead's count of blank lines
16. lOCodeAndComment: numeric
17. uniq_Op         : numeric % unique operators
18. uniq_Opnd       : numeric % unique operands
19. total_Op        : numeric % total operators
20. total_Opnd      : numeric % total operands
21. branchCount     : numeric % of the flow graph
22. problems        : {false,true} % module has/has not one or more reported defects

### Relevant papers  

- Shepperd, M. and Qinbao Song and Zhongbin Sun and Mair, C. (2013)
Data Quality: Some Comments on the NASA Software Defect Datasets, IEEE Transactions on Software Engineering, 39.

- Tim Menzies and Justin S. Di Stefano (2004) How Good is Your Blind Spot Sampling Policy? 2004 IEEE Conference on High Assurance
Software Engineering.

- T. Menzies and J. DiStefano and A. Orrego and R. Chapman (2004) Assessing Predictors of Software Defects", Workshop on Predictive Software Models, Chicago
====
Target Variable: defects (nominal, 2 distinct): ['0', '1']
====
Features:

loc (numeric, 139 distinct): ['2.0', '4.0', '1.0', '3.0', '6.0', '5.0', '15.0', '9.0', '7.0', '16.0']
v(g) (numeric, 31 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '7.0', '6.0', '9.0', '8.0', '11.0']
ev(g) (numeric, 21 distinct): ['1.0', '3.0', '5.0', '4.0', '7.0', '8.0', '6.0', '11.0', '10.0', '9.0']
iv(g) (numeric, 26 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '11.0']
n (numeric, 278 distinct): ['4.0', '5.0', '1.0', '9.0', '0.0', '7.0', '3.0', '8.0', '25.0', '10.0']
v (numeric, 729 distinct): ['8.0', '0.0', '11.61', '4.75', '19.65', '15.51', '27.0', '31.7', '24.0', '28.53']
l (numeric, 52 distinct): ['0.67', '0.0', '0.4', '0.5', '0.33', '0.06', '0.07', '1.0', '0.05', '0.11']
d (numeric, 548 distinct): ['1.5', '0.0', '2.5', '2.0', '3.0', '1.0', '3.5', '6.0', '4.5', '5.0']
i (numeric, 893 distinct): ['5.33', '0.0', '7.74', '4.75', '5.8', '8.0', '7.86', '12.68', '9.51', '7.75']
e (numeric, 961 distinct): ['12.0', '0.0', '17.41', '4.75', '23.22', '79.25', '49.13', '8.0', '31.02', '60.0']
b (numeric, 92 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.08', '0.1', '0.07']
t (numeric, 947 distinct): ['0.67', '0.0', '0.97', '0.26', '1.29', '4.4', '2.73', '0.44', '1.72', '3.33']
lOCode (numeric, 121 distinct): ['0.0', '2.0', '1.0', '3.0', '6.0', '4.0', '5.0', '8.0', '12.0', '9.0']
lOComment (numeric, 28 distinct): ['0', '1', '2', '3', '4', '5', '6', '11', '8', '10']
lOBlank (numeric, 31 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '10']
locCodeAndComment (numeric, 12 distinct): ['0', '1', '2', '3', '4', '5', '7', '11', '6', '8']
uniq_Op (numeric, 34 distinct): ['3.0', '8.0', '5.0', '6.0', '7.0', '1.0', '9.0', '4.0', '10.0', '2.0']
uniq_Opnd (numeric, 73 distinct): ['1.0', '2.0', '0.0', '6.0', '3.0', '4.0', '5.0', '8.0', '7.0', '10.0']
total_Op (numeric, 207 distinct): ['3.0', '1.0', '4.0', '5.0', '2.0', '6.0', '7.0', '0.0', '9.0', '12.0']
total_Opnd (numeric, 153 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '10.0', '5.0', '6.0', '8.0', '7.0']
branchCount (numeric, 44 distinct): ['1.0', '3.0', '5.0', '7.0', '9.0', '11.0', '13.0', '15.0', '17.0', '21.0']
'''

CONTEXT = "Source Code Quality Prediction: KC1"
TARGET = CuratedTarget(raw_name="defects", new_name="Is Defective", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="loc", new_name="Line count of code"),
                CuratedFeature(raw_name="v(g)", new_name="Cyclomatic complexity"),
                CuratedFeature(raw_name="ev(g)", new_name="Essential complexity"),
                CuratedFeature(raw_name="iv(g)", new_name="Design complexity"),
                CuratedFeature(raw_name="n", new_name="total operators + operands"),
                CuratedFeature(raw_name="v", new_name="volume"),
                CuratedFeature(raw_name="l", new_name="program length"),
                CuratedFeature(raw_name="d", new_name="difficulty"),
                CuratedFeature(raw_name="i", new_name="intelligence"),
                CuratedFeature(raw_name="e", new_name="effort"),
                CuratedFeature(raw_name="b"),
                CuratedFeature(raw_name="t", new_name="time estimator"),
                CuratedFeature(raw_name="lOCode", new_name="line count"),
                CuratedFeature(raw_name="lOComment", new_name="count of lines of comments"),
                CuratedFeature(raw_name="lOBlank", new_name="count of blank lines"),
                CuratedFeature(raw_name="locCodeAndComment", new_name="count of code and comment lines"),
                CuratedFeature(raw_name="uniq_Op", new_name="unique operators"),
                CuratedFeature(raw_name="uniq_Opnd", new_name="unique operands"),
                CuratedFeature(raw_name="total_Op", new_name="total operators"),
                CuratedFeature(raw_name="total_Opnd", new_name="total operands"),
                CuratedFeature(raw_name="branchCount", new_name="branch count percentage of the flow graph")]
