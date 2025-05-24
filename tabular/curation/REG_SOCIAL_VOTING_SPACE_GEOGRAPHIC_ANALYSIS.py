from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: space_ga
====
Examples: 3107
====
URL: https://www.openml.org/search?type=data&id=507
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Geographical Analysis Spatial Data

This georeferenced data set was used in:

Pace, R. Kelley, and Ronald Barry, Quick Computation of Regressions with a Spatially
Autoregressive Dependent Variable, Geographical Analysis, Volume 29, Number 3, July
1997, p. 232-247.

It contains 3,107 observations on U.S. county votes cast in the 1980 presidential election.
Specifically, it contains the total number of votes cast in the 1980 presidential election per
county (VOTES), the population in each county of 18 years of age or older (POP), the
population in each county with a 12th grade or higher education (EDUCATION), the
number of owner-occupied housing units (HOUSES), the aggregate income (INCOME), the X
spatial coordinate of the county (XCOORD), and the Y spatial coordinate of the county
(YCOORD).

The dependent variable is the log of the proportion of votes cast for both candidates in the
1980 presidential election. Hence, we can express our dependent variable as ln(VOTES/
POP) = ln(VOTES)-ln(POP).

The overall data set has the following structure

[ln(VOTES/POP) POP EDUCATION HOUSES INCOME XCOORD YCOORD]

Additional details can be found, along with other data, manuscripts, free spatial software, and
so forth, at www.spatial-statistics.com or www.finance.lsu.edu/re (follow the spatial statistics
link). In particular, the above mentioned manuscript which used the data is available for
download. If you have any questions, send an email to kelley@spatial-statistics.com.






Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: 1


Data Description

The dataset contains 3,107 observations on U.S. county votes cast in the 1980 presidential election.

Given population, education levels in the population, number of owned houses, incomes and coordinates, the goal is to predict the number of votes per population (logarithmic).

Attribute Description

ln_votes_pop - logarithm of total number of votes - logarithm of populaiton (ln(votes/pop)), target feature
pop - the population in each county of 18 years of age or older
education - the population in each county with a 12th grade or higher education
houses - the number of owner-occupied housing units
income - the aggregate income
xcoord - X spatial coordinate of the county
ycoord - Y spatial coordinate of the county

====
Target Variable: ln(VOTES/POP) (numeric, 3105 distinct): ['-0.3607', '-0.1671', '-0.6616', '-0.6015', '-0.4003', '-0.5198', '-0.507', '-0.6764', '-0.6963', '-0.4659']
====
Features:

POP (numeric, 3001 distinct): ['8.7614', '8.865', '8.562', '8.1014', '9.5541', '9.9729', '10.0494', '9.9937', '8.5039', '8.4519']
EDUCATION (numeric, 2914 distinct): ['8.1277', '8.1101', '8.2441', '8.3217', '7.6118', '7.8083', '7.4122', '7.4553', '8.6837', '8.6048']
HOUSES (numeric, 2832 distinct): ['8.3907', '8.0346', '7.826', '7.6324', '7.4616', '8.2085', '7.774', '7.8431', '8.3554', '8.633']
INCOME (numeric, 3097 distinct): ['12.1931', '10.9427', '10.6868', '10.5766', '11.385', '10.7643', '11.3729', '10.5812', '10.2288', '11.8686']
XCOORD (numeric, 3107 distinct): ['-86641472.0', '-84167334.0', '-81453275.0', '-83056788.0', '-84577226.0', '-82236075.0', '-83032285.0', '-83072020.0', '-81197599.0', '-84652255.0']
YCOORD (numeric, 3106 distinct): ['39917000.0', '32542207.0', '40525913.0', '39766920.0', '41593455.0', '41117567.0', '39735104.0', '39639560.0', '39083659.0', '41171000.0']
'''

CONTEXT = "US County 1980 Elections Voting by County"
TARGET = CuratedTarget(raw_name='ln(VOTES/POP)', new_name='Logarithm of Votes per Population',
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name='POP', new_name='Population of 18 years or older'),
            CuratedFeature(raw_name='EDUCATION', new_name='Education Proportion with a 12th grade or higher education'),
            CuratedFeature(raw_name='HOUSES', new_name='Number of owner occupied Houses'),
            CuratedFeature(raw_name='INCOME'),
            CuratedFeature(raw_name='XCOORD', new_name='X Coordinate'),
            CuratedFeature(raw_name='YCOORD', new_name='Y Coordinate'), ]