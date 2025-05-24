import numpy as np

from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: pbcseq
====
Examples: 1945
====
URL: https://www.openml.org/search?type=data&id=516
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Primary Biliary Cirrhosis

This data set is a follow-up to the original PBC data set, as discussed
in appendix D of Fleming and Harrington, Counting Processes and Survival
Analysis, Wiley, 1991.  An analysis based on the enclised data is found in
Murtaugh PA. Dickson ER. Van Dam GM. Malinchoc M. Grambsch PM.
Langworthy AL. Gips CH.  "Primary biliary cirrhosis: prediction of short-term
survival based on repeated patient visits." Hepatology. 20(1.1):126-34, 1994.

Quoting from F&H.  "The following pages contain the data from the Mayo Clinic
trial in primary biliary cirrhosis (PBC) of the liver conducted between 1974
and 1984.  A description of the clinical background for the trial and the
covariates recorded here is in Chapter 0, especially Section 0.2.  A more
extended discussion can be found in Dickson, et al., Hepatology 10:1-7 (1989)
and in Markus, et al., N Eng J of Med 320:1709-13 (1989).
"A total of 424 PBC patients, referred to Mayo Clinic during that ten-year
interval, met eligibility criteria for the randomized placebo controlled
trial of the drug D-penicillamine.  The first 312 cases in the data set
participated in the randomized trial and contain largely complete data.  The
additional 112 cases did not participate in the clinical trial, but consented
to have basic measurements recorded and to be followed for survival.  Six of
those cases were lost to follow-up shortly after diagnosis, so the data here
are on an additional 106 cases as well as the 312 randomized participants.
Missing data items are denoted by `.'. "

The F&H data set contains only baseline measurements of the laboratory
paramters.  This data set contains multiple laboratory results, but
only on the first 312 patients.  Some baseline data values in this file
differ from the original PBC file, for instance, the data errors in
prothrombin time and age which were discovered after the orignal analysis,
during research work on dfbeta residuals.  (These two data points are
discussed in F&H, figure 4.6.7).  Another major difference is that
there was significantly more follow-up for many of the patients at the
time this data set was assembled.

One "feature" of the data deserves special comment.  The last
observation before death or liver transplant often has many more
missing covariates than other data rows.  The original clinical
protocol for these patients specified visits at 6 months, 1 year, and
annually thereafter.  At these protocol visits lab values were
obtained for a large pre-specified battery of tests.  "Extra" visits,
often undertaken because of worsening medical condition, did not
necessarily have all this lab work.  The missing values are thus
potentially informative, and violate the usual "missing at random"
(MCAR or MAC) assumptions that are assumed in analyses.  Because of
the earlier published results on the Mayo PBC risk score, however, the
5 variables involved in that computation were usually obtained, i.e.,
age, bilirubin, albumin, prothrombin time, and edema score.

Variables:
case number
number of days between registration and the earlier of death,
transplantion, or study analysis time
status: 0=alive, 1=transplanted, 2=dead
drug: 1= D-penicillamine, 0=placebo
age in days, at registration
sex: 0=male, 1=female
day: number of days between enrollment and this visit date, remaining
values on the line of data refer to this visit.
presence of asictes:       0=no 1=yes
presence of hepatomegaly   0=no 1=yes
presence of spiders        0=no 1=yes
presence of edema          0=no edema and no diuretic therapy for edema;
.5 = edema present without diuretics, or edema resolved by diuretics;
1 = edema despite diuretic therapy
serum bilirubin in mg/dl
serum cholesterol in mg/dl
albumin in gm/dl
alkaline phosphatase in U/liter
SGOT in U/ml  (serum glutamic-oxaloacetic transaminase, the enzyme name
has subsequently changed to "ALT" in the medical literature)
platelets per cubic ml / 1000
prothrombin time in seconds
histologic stage of disease


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: 3
====
Target Variable: status (numeric, 3 distinct): ['0', '2', '1']
====
Features:

case_number (numeric, 312 distinct): ['58', '32', '42', '40', '43', '93', '19', '83', '73', '34']
number_of_days (numeric, 305 distinct): ['3086', '5192', '5128', '5122', '5225', '4901', '4859', '5136', '4719', '4583']
drug (nominal, 2 distinct): ['D-penicillamine', '0']
age (numeric, 308 distinct): ['17841', '16279', '12307', '22960', '19722', '18102', '17850', '13344', '17046', '14060']
sex (nominal, 2 distinct): ['female', 'male']
day (nominal, 1024 distinct): ['no', '182', '179', '376', '184', '185', '1098', '180', '183', '188']
presence_of_asictes (nominal, 3 distinct): ['no', 'yes']
presence_of_hepatomegaly (nominal, 3 distinct): ['no', 'yes']
presence_of_spiders (nominal, 3 distinct): ['0', '1']
presence_of_edema (numeric, 3 distinct): ['0.0', '0.5', '1.0']
serum_bilirubin (numeric, 193 distinct): ['0.6', '0.5', '0.7', '0.8', '0.9', '1.1', '1.0', '1.2', '1.3', '0.4']
serum_cholesterol (numeric, 376 distinct): ['260.0', '296.0', '246.0', '244.0', '258.0', '239.0', '325.0', '250.0', '219.0', '257.0']
albumin (numeric, 254 distinct): ['3.6', '3.46', '3.5', '3.35', '3.7', '3.56', '3.2', '3.55', '3.37', '3.61']
alkaline_phosphatase (numeric, 1264 distinct): ['1119.0', '464.0', '674.0', '1052.0', '705.0', '836.0', '814.0', '1498.0', '938.0', '996.0']
SGOT (numeric, 418 distinct): ['76.0', '107.0', '170.5', '45.0', '116.3', '99.2', '68.2', '53.0', '119.4', '124.0']
platelets (numeric, 415 distinct): ['272.0', '204.0', '166.0', '264.0', '250.0', '247.0', '283.0', '200.0', '268.0', '181.0']
prothrombin_time (numeric, 78 distinct): ['10.6', '11.0', '10.1', '10.0', '10.8', '9.9', '10.7', '10.5', '10.9', '10.4']
histologic_stage_of_disease (numeric, 4 distinct): ['4', '3', '2', '1']
'''

CONTEXT = "Clinical data for Primary Biliary Cirrhosis patients"
TARGET = CuratedTarget(raw_name="status", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"0": "Alive", "1": "Transplanted", "2": "Dead"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="day", feat_type=FeatureType.NUMERIC, numeric_missing="no")]