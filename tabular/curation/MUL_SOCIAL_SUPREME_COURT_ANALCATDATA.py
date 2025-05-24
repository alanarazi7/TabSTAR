from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: analcatdata_supreme
====
Examples: 4052
====
URL: https://www.openml.org/search?type=data&id=504
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

analcatdata    A collection of data sets used in the book "Analyzing Categorical Data,"
by Jeffrey S. Simonoff, Springer-Verlag, New York, 2003. The submission
consists of a zip file containing two versions of each of 84 data sets,
plus this README file. Each data set is given in comma-delimited ASCII
(.csv) form, and Microsoft Excel (.xls) form.

NOTICE: These data sets may be used freely for scientific, educational and/or
noncommercial purposes, provided suitable acknowledgment is given (by citing
the above-named reference).

Further details concerning the book, including information on statistical software
(including sample S-PLUS/R and SAS code), are available at the web site

http://www.stern.nyu.edu/~jsimonof/AnalCatData


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific


Note: Quotes, Single-Quotes and Backslashes were removed, Blanks replaced
with Underscores
====
Target Variable: Log_exposure (numeric, 10 distinct): ['2.3', '1.79', '1.39', '0.69', '1.61', '1.95', '1.1', '0.0', '2.2', '2.08']
====
Features:

Actions_taken (numeric, 10 distinct): ['0', '1', '2', '3', '6', '5', '4', '7', '11', '10']
Liberal (numeric, 2 distinct): ['1', '0']
Unconstitutional (numeric, 2 distinct): ['0', '1']
Precedent_alteration (numeric, 2 distinct): ['0', '1']
Unanimous (numeric, 2 distinct): ['0', '1']
Year_of_decision (numeric, 36 distinct): ['1976.0', '1974.0', '1984.0', '1973.0', '1985.0', '1983.0', '1982.0', '1977.0', '1987.0', '1971.0']
Lower_court_disagreement (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Supreme Court Cases"
TARGET = CuratedTarget(raw_name="Log_exposure", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []