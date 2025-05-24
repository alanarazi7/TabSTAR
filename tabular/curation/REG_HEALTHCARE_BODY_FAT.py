from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: bodyfat
====
Examples: 252
====
URL: https://www.openml.org/search?type=data&id=560
====
Description: **Author**: Roger W. Johnson  
**Source**: [UCI (not available anymore)](https://archive.ics.uci.edu/ml/index.php), [TunedIT](http://tunedit.org/repo/UCI/numeric/bodyfat.arff)  
**Please cite**: None. 

Short Summary:
Lists estimates of the percentage of body fat determined by underwater
weighing and various body circumference measurements for 252 men.

Classroom use of this data set:
This data set can be used to illustrate multiple regression techniques.
Accurate measurement of body fat is inconvenient/costly and it is
desirable to have easy methods of estimating body fat that are not
inconvenient/costly.

More Details:
A variety of popular health books suggest that the readers assess their
health, at least in part, by estimating their percentage of body fat. In
Bailey (1994), for instance, the reader can estimate body fat from tables
using their age and various skin-fold measurements obtained by using a
caliper. Other texts give predictive equations for body fat using body
circumference measurements (e.g. abdominal circumference) and/or skin-fold
measurements. See, for instance, Behnke and Wilmore (1974), pp. 66-67;
Wilmore (1976), p. 247; or Katch and McArdle (1977), pp. 120-132).

Percentage of body fat for an individual can be estimated once body density
has been determined. Folks (e.g. Siri (1956)) assume that the body consists
of two components - lean body tissue and fat tissue. Letting

D = Body Density (gm/cm^3)
A = proportion of lean body tissue
B = proportion of fat tissue (A+B=1)
a = density of lean body tissue (gm/cm^3)
b = density of fat tissue (gm/cm^3)

we have

D = 1/[(A/a) + (B/b)]

solving for B we find

B = (1/D)*[ab/(a-b)] - [b/(a-b)].

Using the estimates a=1.10 gm/cm^3 and b=0.90 gm/cm^3 (see Katch and McArdle
(1977), p. 111 or Wilmore (1976), p. 123) we come up with "Siri's equation":

Percentage of Body Fat (i.e. 100*B) = 495/D - 450.

Volume, and hence body density, can be accurately measured a variety of ways.
The technique of underwater weighing "computes body volume as the difference
between body weight measured in air and weight measured during water
submersion. In other words, body volume is equal to the loss of weight in
water with the appropriate temperature correction for the water's density"
(Katch and McArdle (1977), p. 113). Using this technique,

Body Density = WA/[(WA-WW)/c.f. - LV]

where

WA = Weight in air (kg)
WW = Weight in water (kg)
c.f. = Water correction factor (=1 at 39.2 deg F as one-gram of water
occupies exactly one cm^3 at this temperature, =.997 at 76-78 deg F)
LV = Residual Lung Volume (liters)

(Katch and McArdle (1977), p. 115). Other methods of determining body volume
are given in Behnke and Wilmore (1974), p. 22 ff.


The variables listed below, from left to right, are:

Density determined from underwater weighing
Percent body fat from Siri's (1956) equation
Age (years)
Weight (lbs)
Height (inches)
Neck circumference (cm)
Chest circumference (cm)
Abdomen 2 circumference (cm)
Hip circumference (cm)
Thigh circumference (cm)
Knee circumference (cm)
Ankle circumference (cm)
Biceps (extended) circumference (cm)
Forearm circumference (cm)
Wrist circumference (cm)

(Measurement standards are apparently those listed in Benhke and Wilmore
(1974), pp. 45-48 where, for instance, the abdomen 2 circumference is
measured "laterally, at the level of the iliac crests, and anteriorly, at
the umbilicus".)

These data are used to produce the predictive equations for lean
body weight given in the abstract "Generalized body composition prediction
equation for men using simple measurement techniques", K.W. Penrose, A.G.
Nelson, A.G. Fisher, FACSM, Human Performance Research Center, Brigham Young
University, Provo, Utah  84602 as listed in _Medicine and Science in Sports
and Exercise_, vol. 17, no. 2, April 1985, p. 189. (The predictive equations
were obtained from the first 143 of the 252 cases that are listed below).
The data were generously supplied by Dr. A. Garth Fisher who gave permission to
freely distribute the data and use for non-commercial purposes.

References:

Bailey, Covert (1994). _Smart Exercise: Burning Fat, Getting Fit_,
Houghton-Mifflin Co., Boston, pp. 179-186.

Behnke, A.R. and Wilmore, J.H. (1974). _Evaluation and Regulation of Body
Build and Composition_, Prentice-Hall, Englewood Cliffs, N.J.

Siri, W.E. (1956), "Gross composition of the body", in _Advances in
Biological and Medical Physics_, vol. IV, edited by J.H. Lawrence and C.A.
Tobias, Academic Press, Inc., New York.

Katch, Frank and McArdle, William (1977). _Nutrition, Weight Control, and
Exercise_, Houghton Mifflin Co., Boston.

Wilmore, Jack (1976). _Athletic Training and Physical Fitness: Physiological
Principles of the Conditioning Process_, Allyn and Bacon, Inc., Boston.
====
Target Variable: class (numeric, 176 distinct): ['20.4', '25.8', '23.6', '14.9', '16.5', '22.1', '20.5', '8.8', '9.4', '19.2']
====
Features:

Density (numeric, 218 distinct): ['1.061', '1.0414', '1.0484', '1.0524', '1.0462', '1.0424', '1.0775', '1.079', '1.0648', '1.0373']
Age (numeric, 51 distinct): ['40', '43', '42', '47', '55', '41', '35', '44', '49', '54']
Weight (numeric, 197 distinct): ['179.75', '168.0', '172.75', '167.0', '177.25', '170.75', '184.25', '152.25', '168.25', '161.75']
Height (numeric, 48 distinct): ['71.5', '69.25', '72.25', '69.5', '67.5', '70.0', '69.75', '67.75', '68.25', '70.5']
Neck (numeric, 90 distinct): ['38.5', '38.0', '37.4', '37.8', '36.5', '38.7', '40.8', '35.5', '37.5', '42.1']
Chest (numeric, 174 distinct): ['102.7', '99.1', '97.8', '99.6', '94.0', '98.9', '92.3', '102.0', '93.5', '104.0']
Abdomen (numeric, 185 distinct): ['88.7', '100.5', '89.7', '95.0', '83.6', '98.6', '99.8', '105.0', '95.6', '92.4']
Hip (numeric, 152 distinct): ['98.3', '100.6', '99.3', '96.2', '94.5', '102.5', '94.0', '96.1', '96.9', '101.7']
Thigh (numeric, 139 distinct): ['58.9', '58.5', '56.0', '57.5', '54.7', '59.3', '60.6', '63.5', '57.1', '59.1']
Knee (numeric, 90 distinct): ['39.0', '37.3', '38.1', '38.3', '38.0', '36.2', '40.0', '38.7', '38.4', '39.4']
Ankle (numeric, 61 distinct): ['22.0', '22.5', '22.6', '21.8', '23.2', '22.7', '23.4', '21.5', '22.4', '21.9']
Biceps (numeric, 104 distinct): ['31.6', '30.5', '32.5', '30.1', '31.0', '31.4', '30.3', '33.6', '27.9', '33.5']
Forearm (numeric, 77 distinct): ['27.3', '29.8', '29.6', '29.2', '27.4', '26.3', '30.0', '30.1', '28.4', '28.2']
Wrist (numeric, 44 distinct): ['18.8', '18.5', '17.7', '18.4', '18.2', '17.4', '18.3', '19.0', '17.6', '17.3']
'''

CONTEXT = "Body Fat Percentage for men"
TARGET = CuratedTarget(raw_name="class", new_name="Body Fat", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []