from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: analcatdata_authorship
====
Examples: 841
====
URL: https://www.openml.org/search?type=data&id=458
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
CLASSTYPE: nominal
CLASSINDEX: last


Note: Quotes, Single-Quotes and Backslashes were removed, Blanks replaced
with Underscores
====
Target Variable: Author (nominal, 4 distinct): ['Austen', 'London', 'Shakespeare', 'Milton']
====
Features:

a (numeric, 57 distinct): ['30', '28', '33', '25', '29', '27', '31', '36', '24', '35']
all (numeric, 27 distinct): ['8', '7', '6', '9', '10', '5', '4', '11', '12', '13']
also (numeric, 6 distinct): ['0', '1', '2', '3', '4', '6']
an (numeric, 50 distinct): ['4', '3', '2', '5', '6', '1', '0', '7', '8', '9']
and (numeric, 83 distinct): ['51', '58', '46', '55', '53', '45', '44', '47', '54', '61']
any (numeric, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
are (numeric, 21 distinct): ['0', '1', '2', '3', '4', '5', '7', '6', '8', '9']
as (numeric, 31 distinct): ['11', '8', '12', '13', '14', '15', '16', '10', '9', '7']
at (numeric, 22 distinct): ['8', '6', '10', '9', '11', '7', '5', '12', '4', '3']
be (numeric, 39 distinct): ['6', '7', '5', '4', '8', '17', '13', '9', '16', '3']
been (numeric, 24 distinct): ['2', '1', '0', '3', '5', '4', '8', '6', '7', '9']
but (numeric, 25 distinct): ['11', '10', '12', '13', '15', '9', '14', '8', '16', '7']
by (numeric, 21 distinct): ['6', '8', '7', '10', '5', '9', '11', '4', '3', '12']
can (numeric, 12 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '10', '8']
do (numeric, 22 distinct): ['2', '0', '1', '3', '4', '5', '6', '7', '9', '8']
down (numeric, 13 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
even (numeric, 9 distinct): ['0', '1', '2', '3', '4', '5', '7', '9', '6']
every (numeric, 12 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for (numeric, 27 distinct): ['13', '12', '11', '15', '14', '10', '17', '16', '18', '9']
from (numeric, 25 distinct): ['5', '7', '6', '4', '8', '3', '9', '2', '10', '11']
had (numeric, 45 distinct): ['1', '2', '3', '7', '11', '4', '10', '14', '13', '16']
has (numeric, 15 distinct): ['0', '1', '2', '3', '4', '5', '8', '6', '7', '10']
have (numeric, 31 distinct): ['2', '8', '3', '9', '6', '7', '5', '4', '11', '12']
her (numeric, 66 distinct): ['0', '1', '4', '2', '6', '3', '5', '13', '8', '10']
his (numeric, 54 distinct): ['10', '8', '9', '12', '11', '13', '14', '7', '6', '16']
if (numeric, 18 distinct): ['3', '1', '2', '4', '5', '6', '0', '7', '8', '9']
in (numeric, 45 distinct): ['27', '28', '22', '26', '25', '23', '29', '30', '24', '20']
into (numeric, 15 distinct): ['2', '1', '3', '0', '4', '5', '6', '7', '8', '9']
is (numeric, 40 distinct): ['1', '0', '5', '2', '12', '3', '4', '13', '7', '9']
it (numeric, 49 distinct): ['20', '18', '22', '21', '16', '23', '27', '17', '14', '19']
its (numeric, 12 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
may (numeric, 11 distinct): ['0', '1', '2', '3', '4', '6', '5', '7', '9', '8']
more (numeric, 15 distinct): ['3', '4', '2', '5', '6', '1', '7', '8', '0', '9']
must (numeric, 16 distinct): ['2', '0', '1', '3', '4', '5', '6', '7', '8', '9']
my (numeric, 51 distinct): ['0', '3', '2', '5', '7', '1', '4', '10', '8', '6']
no (numeric, 23 distinct): ['4', '6', '5', '3', '8', '7', '9', '10', '2', '11']
not (numeric, 37 distinct): ['15', '12', '16', '19', '18', '8', '17', '13', '14', '9']
now (numeric, 16 distinct): ['2', '3', '4', '1', '5', '0', '6', '8', '7', '9']
of (numeric, 76 distinct): ['41', '46', '39', '45', '53', '50', '37', '55', '49', '33']
on (numeric, 26 distinct): ['8', '6', '9', '7', '11', '5', '12', '10', '4', '13']
one (numeric, 18 distinct): ['3', '4', '2', '5', '1', '6', '7', '8', '0', '9']
only (numeric, 11 distinct): ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
or (numeric, 29 distinct): ['2', '4', '3', '5', '6', '1', '7', '0', '8', '9']
our (numeric, 30 distinct): ['0', '1', '2', '3', '5', '4', '6', '7', '8', '13']
should (numeric, 14 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
so (numeric, 24 distinct): ['7', '8', '5', '9', '6', '10', '4', '3', '11', '2']
some (numeric, 13 distinct): ['2', '1', '3', '0', '4', '5', '6', '7', '8', '9']
such (numeric, 14 distinct): ['1', '3', '2', '0', '4', '5', '6', '7', '8', '10']
than (numeric, 14 distinct): ['2', '3', '4', '1', '5', '0', '6', '7', '8', '9']
that (numeric, 38 distinct): ['17', '21', '19', '16', '18', '14', '20', '22', '15', '26']
the (numeric, 137 distinct): ['61', '53', '69', '70', '57', '80', '62', '56', '46', '50']
their (numeric, 26 distinct): ['2', '3', '1', '4', '5', '0', '6', '7', '8', '9']
then (numeric, 13 distinct): ['2', '3', '1', '4', '0', '5', '6', '7', '8', '9']
there (numeric, 16 distinct): ['5', '4', '3', '6', '2', '7', '1', '8', '9', '10']
things (numeric, 10 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '11', '7']
this (numeric, 27 distinct): ['5', '6', '4', '7', '3', '9', '8', '2', '10', '11']
to (numeric, 63 distinct): ['40', '35', '38', '33', '37', '36', '41', '34', '53', '55']
up (numeric, 14 distinct): ['2', '1', '3', '4', '0', '5', '6', '7', '8', '9']
upon (numeric, 11 distinct): ['1', '0', '2', '3', '4', '5', '6', '7', '8', '10']
was (numeric, 64 distinct): ['2', '1', '3', '20', '4', '26', '5', '27', '34', '29']
were (numeric, 31 distinct): ['2', '1', '4', '3', '5', '8', '6', '7', '9', '10']
what (numeric, 21 distinct): ['4', '5', '3', '7', '6', '2', '8', '1', '9', '10']
when (numeric, 16 distinct): ['4', '3', '5', '2', '6', '7', '1', '8', '9', '10']
which (numeric, 20 distinct): ['3', '4', '2', '5', '1', '6', '7', '8', '0', '9']
who (numeric, 16 distinct): ['2', '3', '1', '4', '0', '5', '6', '7', '8', '9']
will (numeric, 25 distinct): ['0', '1', '3', '2', '5', '4', '6', '7', '8', '9']
with (numeric, 36 distinct): ['14', '12', '15', '11', '13', '16', '10', '17', '8', '9']
would (numeric, 23 distinct): ['2', '3', '1', '5', '4', '6', '0', '7', '8', '9']
your (numeric, 31 distinct): ['0', '1', '3', '2', '4', '5', '7', '9', '8', '6']
BookID (numeric, 12 distinct): ['1', '6', '4', '5', '2', '7', '3', '10', '8', '11']
'''

CONTEXT = "Book Authorship Prediction Based on Word Cnt Usage"
TARGET = CuratedTarget(raw_name="Author", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="BookID", feat_type=FeatureType.CATEGORICAL)]