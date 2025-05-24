from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cars
====
Examples: 406
====
URL: https://www.openml.org/search?type=data&id=455
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

The Committee on Statistical Graphics of the American Statistical
Association (ASA) invites you to participate in its Second (1983)
Exposition of Statistical Graphics Technology. The purposes of the
Exposition are (l) to provide a forum in which users and providers of
statistical graphics technology can exchange information and ideas and
(2) to expose those members of the ASA community who are less familiar
with statistical graphics to its capabilities and potential benefits
to them. The Exposition wil1 be held in conjunction with the Annual
Meetings in Toronto, August 15-18, 1983 and is tentatively scheduled
for the afternoon of Wednesday, August 17.

Seven providers of statistical graphics technology participated in the
l982 Exposition. By all accounts, the Exposition was well received by
the ASA community and was a worthwhile experience for the
participants. We hope to have those seven involved again this year,
along with as many new participants as we can muster. The 1982
Exposition was summarized in a paper to appear in the Proceeding of
the Statistical Computing Section. A copy of that paper is enclosed
for your information.

The basic format of the 1983 Exposition will be similar to that of
1982. However, based upon comments received and experience gained,
there are some changes. The basic structure, intended to be simpler
and more flexible than last year, is as follows:

A fixed data set is to be analyzed. This data set is a version of the
CRCARS data set of

Donoho, David and Ramos, Ernesto (1982), ``PRIMDATA:
Data Sets for Use With PRIM-H'' (DRAFT).

Because of the Committee's limited (zero) budget for the Exposition,
we are forced to provide the data in hardcopy form only (enclosed).
(Sorry!)

There are 406 observations on the following 8 variables: MPG (miles
per gallon), # cylinders, engine displacement (cu. inches), horsepower,
vehicle weight (lbs.), time to accelerate from O to 60 mph (sec.),
model year (modulo 100), and origin of car (1. American, 2. European,
3. Japanese). These data appear on seven pages. Also provided are the
car labels (types) in the same order as the 8 variables on seven
separate pages. Missing data values are marked by series of question
marks.

You are asked to analyze these data using your statistical graphics
software. Your objective should be to achieve graphical displays which
will be meaningful to the viewers and highlight relevant aspects of
the data. If you can best achieve this using simple graphical formats,
fine. If you choose to illustrate some of the more sophisticated
capabilities of your software and can do so without losing relevancy
to the data, that is fine, too. This year, there will be no Committee
commentary on the individual presentations, so you are not competing
with other presenters. The role of each presenter is to do his/her
best job of presenting their statistical graphics technology to the
viewers.

Each participant will be provided with a 6'(long) by 4'(tall)
posterboard on which to display the results of their analyses. This is
the same format as last year. You are encouraged to remain by your
presentation during the Exposition to answer viewers' questions. Three
copies of your presentation must be submitted to me by July 1. Movie
or slide show presentations cannot be accommodated (sorry). The
Committee will prepare its own poster presentation which will orient
the viewers to the data and the purposes of the Exposition.

The ASA has asked us to remind all participants that the Exposition is
intended for educational and scientific purposes and is not a
marketing activity. Even though last year's participants did an
excellent job of maintaining that distinction, a cautionary note at
this point is appropriate.

Those of us who were involved with the 1982 Exposition found it
worthwhile and fun to do. We would very much like to have you
participate this year. For planning purposes, please RSVP (to me, in
writing please) by April 15 as to whether you plan to accept the
Committee's invitation.

If you have any questions about the Exposition, please call me on
(301/763-5350). If you have specific questions about the data, or the
analysis, please call Karen Kafadar on (301/921-3651). If you cannot
participate but know of another person or group in your organization
who can, please pass this invitation along to them.

Sincerely,



LAWRENCE H. COX
Statistical Research Division
Bureau of the Census
Room 3524-3
Washington, DC 20233




Information about the dataset
CLASSTYPE: nominal
CLASSINDEX: last
====
Target Variable: origin (nominal, 3 distinct): ['1', '3', '2']
====
Features:

mpg (numeric, 130 distinct): ['13.0', '14.0', '18.0', '15.0', '26.0', '16.0', '19.0', '25.0', '24.0', '22.0']
cylinders (nominal, 5 distinct): ['4', '8', '6', '3', '5']
displacement (numeric, 83 distinct): ['97.0', '350.0', '98.0', '250.0', '318.0', '140.0', '400.0', '225.0', '91.0', '121.0']
horsepower (numeric, 94 distinct): ['150.0', '90.0', '88.0', '110.0', '100.0', '95.0', '75.0', '70.0', '105.0', '67.0']
weight (numeric, 356 distinct): ['2130', '1985', '2300', '2720', '2155', '2125', '2265', '2945', '3672', '2950']
acceleration (numeric, 96 distinct): ['14.5', '15.5', '14.0', '16.0', '13.5', '15.0', '17.0', '16.5', '19.0', '13.0']
model.year (numeric, 13 distinct): ['73', '78', '70', '76', '82', '75', '81', '71', '79', '80']
'''

CONTEXT = "Car Origin Country"
TARGET = CuratedTarget(raw_name="origin", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'American', '2': 'European', '3': 'Japanese'})
COLS_TO_DROP = []
FEATURES = []