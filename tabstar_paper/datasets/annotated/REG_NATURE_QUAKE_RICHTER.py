from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: quake
====
Examples: 2178
====
URL: https://www.openml.org/search?type=data&id=550
====

[Duplicate of:
https://www.openml.org/search?type=data&status=active&id=209&sort=runs
with the correct variable names]

Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

File README
-----------

smoothmeth  A collection of the data sets used in the book "Smoothing
Methods in Statistics," by Jeffrey S. Simonoff,
Springer-Verlag, New York, 1996. Submitted by Jeff
Simonoff (jsimonoff@stern.nyu.edu).


This submission consists of 37 files, plus this README file.
Each file represents a data set analyzed in the book.
The names of the files correspond to the names given in
the book. The data files are written in plain ASCII (character)
text. Missing values are represented by "M" in all data files.

Several of the files include an alphabetic (labeling) variable. It is
likely that these files would have to be input into a package using fixed,
rather than free, format. The relevant files, along with appropriate
Fortran format statements, are as follows:

adptvisa.dat: (f10.4,4x,f7.4,3x,a20)

airaccid.dat: (i3,3x,a34)

basesal.dat : (f8.1,4x,a17)

baskball.dat: (f7.4,4x,f6.4,3x,i3,4x,f5.2,3x,i2,3x,a17)

cars93.dat  : (f5.1,2x,i2,2x,i2,3x,f3.1,2x,i3,3x,f4.1,2x,i4,2x,a21)

elusage.dat : (i4,3x,f7.3,2x,a7)

hckshoot.dat: (f7.3,4x,i1,4x,a20)

jantemp.dat : (i6,3x,a30)

marathon.dat: (f10.2,4x,a27)

newscirc.dat: (f8.2,3x,f7.2,2x,a25)

racial.dat  : (f7.4,4x,a32)

safewatr.dat: (i5,3x,i3,3x,a26)

schlvote.dat: (i3,4x,f5.2,2x,i8,4x,f4.1,5x,f5.2,2x,i7,2x,a25)

Description of data sources, and further information about the data sets,
can be found in the "Descriptions of the data sets" section of the book.
Pointing a World Wide Web browser to the URL

http://www.stern.nyu.edu/SOR/SmoothMeth

will provide access to a site devoted to the book.

NOTICE: These datasets may be used freely for scientific,
educational and/or non-commercial purposes, provided suitable
acknowledgment is given (by citing the reference above).


File: ../data/smoothmeth/quake.dat


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific

4 Features
Feature Name	Type	Distinct/Missing Values	Ontology
richter (target)	numeric	12 distinct values
0 missing attributes	
focal_depth	numeric	312 distinct values
0 missing attributes	
latitude	numeric	1824 distinct values
0 missing attributes	
longitude	numeric	1958 distinct values
0 missing attributes	

====
Target Variable: col_4 (numeric, 12 distinct): ['5.8', '5.9', '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7']
====
Features:

col_1 (numeric, 312 distinct): ['0', '33', '10', '45', '43', '36', '46', '37', '44', '38']
col_2 (numeric, 1824 distinct): ['49.88', '49.91', '49.87', '49.92', '-6.55', '-21.86', '-22.91', '73.39', '42.48', '-5.13']
col_3 (numeric, 1958 distinct): ['126.65', '78.92', '78.86', '78.97', '151.86', '78.98', '122.61', '126.02', '148.77', '79.03']
'''

CONTEXT = "Earthquake Richter Scale Prediction"
TARGET = CuratedTarget(raw_name="col_4", new_name="Richter", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="col_1", new_name="Focal Depth"),
            CuratedFeature(raw_name="col_2", new_name="Latitude"),
            CuratedFeature(raw_name="col_3", new_name="Longitude")]