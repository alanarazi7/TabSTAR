from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: hypothyroid
====
Examples: 3772
====
URL: https://www.openml.org/search?type=data&id=57
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

;
 ; Thyroid disease records supplied by the Garavan Institute and J. Ross
 ; Quinlan, New South Wales Institute, Syndney, Australia.
 ;
 ; 1987.
 ;
 
 hypothyroid, primary hypothyroid, compensated hypothyroid,
 secondary hypothyroid,
 negative.			|  classes
 
 age:				continuous.
 sex:				M, F.
 on thyroxine:			f, t.
 query on thyroxine:		f, t.
 on antithyroid medication:	f, t.
 sick:				f, t.
 pregnant:			f, t.
 thyroid surgery:		f, t.
 I131 treatment:			f, t.
 query hypothyroid:		f, t.
 query hyperthyroid:		f, t.
 lithium:			f, t.
 goitre:				f, t.
 tumor:				f, t.
 hypopituitary:			f, t.
 psych:				f, t.
 TSH measured:			f, t.
 TSH:				continuous.
 T3 measured:			f, t.
 T3:				continuous.
 TT4 measured:			f, t.
 TT4:				continuous.
 T4U measured:			f, t.
 T4U:				continuous.
 FTI measured:			f, t.
 FTI:				continuous.
 TBG measured:			f, t.
 TBG:				continuous.
 referral source:		WEST, STMW, SVHC, SVI, SVHD, other.


 Num Instances:     3772
 Num Attributes:    30
 Num Continuous:    7 (Int 1 / Real 6)
 Num Discrete:      23
 Missing values:    6064 /  5.4%

     name                      type enum ints real     missing    distinct  (1)
   1 'age'                     Int    0% 100%   0%     1 /  0%    93 /  2%   0% 
   2 'sex'                     Enum  96%   0%   0%   150 /  4%     2 /  0%   0% 
   3 'on thyroxine'            Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
   4 'query on thyroxine'      Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
   5 'on antithyroid medicati  Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
   6 'sick'                    Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
   7 'pregnant'                Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
   8 'thyroid surgery'         Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
   9 'I131 treatment'          Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  10 'query hypothyroid'       Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  11 'query hyperthyroid'      Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  12 'lithium'                 Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  13 'goitre'                  Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  14 'tumor'                   Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  15 'hypopituitary'           Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  16 'psych'                   Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  17 'TSH measured'            Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  18 'TSH'                     Real   0%  11%  79%   369 / 10%   287 /  8%   2% 
  19 'T3 measured'             Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  20 'T3'                      Real   0%   9%  71%   769 / 20%    69 /  2%   0% 
  21 'TT4 measured'            Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  22 'TT4'                     Real   0%  94%   0%   231 /  6%   241 /  6%   1% 
  23 'T4U measured'            Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  24 'T4U'                     Real   0%   2%  87%   387 / 10%   146 /  4%   1% 
  25 'FTI measured'            Enum 100%   0%   0%     0 /  0%     2 /  0%   0% 
  26 'FTI'                     Real   0%  90%   0%   385 / 10%   234 /  6%   2% 
  27 'TBG measured'            Enum 100%   0%   0%     0 /  0%     1 /  0%   0% 
  28 'TBG'                     Real   0%   0%   0%  3772 /100%     0 /  0%   0% 
  29 'referral source'         Enum 100%   0%   0%     0 /  0%     5 /  0%   0% 
  30 'Class'                   Enum 100%   0%   0%     0 /  0%     4 /  0%   0%
====
Target Variable: Class (nominal, 4 distinct): ['negative', 'compensated_hypothyroid', 'primary_hypothyroid', 'secondary_hypothyroid']
====
Features:

age (numeric, 94 distinct): ['59.0', '60.0', '70.0', '73.0', '55.0', '63.0', '72.0', '58.0', '62.0', '61.0']
sex (nominal, 3 distinct): ['F', 'M']
on_thyroxine (nominal, 2 distinct): ['f', 't']
query_on_thyroxine (nominal, 2 distinct): ['f', 't']
on_antithyroid_medication (nominal, 2 distinct): ['f', 't']
sick (nominal, 2 distinct): ['f', 't']
pregnant (nominal, 2 distinct): ['f', 't']
thyroid_surgery (nominal, 2 distinct): ['f', 't']
I131_treatment (nominal, 2 distinct): ['f', 't']
query_hypothyroid (nominal, 2 distinct): ['f', 't']
query_hyperthyroid (nominal, 2 distinct): ['f', 't']
lithium (nominal, 2 distinct): ['f', 't']
goitre (nominal, 2 distinct): ['f', 't']
tumor (nominal, 2 distinct): ['f', 't']
hypopituitary (nominal, 2 distinct): ['f', 't']
psych (nominal, 2 distinct): ['f', 't']
TSH_measured (nominal, 2 distinct): ['t', 'f']
TSH (numeric, 288 distinct): ['0.2', '1.3', '1.1', '1.4', '1.5', '1.2', '1.9', '1.6', '1.7', '2.3']
T3_measured (nominal, 2 distinct): ['t', 'f']
T3 (numeric, 70 distinct): ['2.0', '1.8', '2.2', '1.9', '2.1', '2.3', '1.6', '1.7', '1.5', '2.4']
TT4_measured (nominal, 2 distinct): ['t', 'f']
TT4 (numeric, 242 distinct): ['101.0', '93.0', '98.0', '103.0', '102.0', '87.0', '91.0', '94.0', '99.0', '95.0']
T4U_measured (nominal, 2 distinct): ['t', 'f']
T4U (numeric, 147 distinct): ['0.99', '0.9', '1.01', '1.0', '0.92', '0.97', '0.93', '1.02', '0.91', '0.95']
FTI_measured (nominal, 2 distinct): ['t', 'f']
FTI (numeric, 235 distinct): ['100.0', '93.0', '114.0', '98.0', '107.0', '92.0', '104.0', '106.0', '96.0', '97.0']
TBG_measured (nominal, 1 distinct): ['f']
TBG (numeric, 1 distinct): []
referral_source (nominal, 5 distinct): ['other', 'SVI', 'SVHC', 'STMW', 'SVHD']
'''

CONTEXT = "Hypothyroidism Patients"
TARGET = CuratedTarget(raw_name="Class", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []