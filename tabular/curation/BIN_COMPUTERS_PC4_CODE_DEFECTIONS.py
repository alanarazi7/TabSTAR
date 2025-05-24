from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: pc4
====
Examples: 1458
====
URL: https://www.openml.org/search?type=data&id=1049
====
Description: **Author**: Mike Chapman, NASA  
**Source**: [tera-PROMISE](http://openscience.us/repo/defect/mccabehalsted/pc1.html) - 2004  
**Please cite**: Sayyad Shirabad, J. and Menzies, T.J. (2005) The PROMISE Repository of Software Engineering Databases. School of Information Technology and Engineering, University of Ottawa, Canada.  
  
**PC4 Software defect prediction**  
One of the NASA Metrics Data Program defect data sets. Data from flight software for earth orbiting satellite. Data comes from McCabe and Halstead features extractors of source code.  These features were defined in the 70s in an attempt to objectively characterize code features that are associated with software quality.

### Relevant papers  

- Shepperd, M. and Qinbao Song and Zhongbin Sun and Mair, C. (2013)
Data Quality: Some Comments on the NASA Software Defect Datasets, IEEE Transactions on Software Engineering, 39.

- Tim Menzies and Justin S. Di Stefano (2004) How Good is Your Blind Spot Sampling Policy? 2004 IEEE Conference on High Assurance
Software Engineering.

- T. Menzies and J. DiStefano and A. Orrego and R. Chapman (2004) Assessing Predictors of Software Defects", Workshop on Predictive Software Models, Chicago
====
Target Variable: c (nominal, 2 distinct): ['0', '1']
====
Features:

LOC_BLANK (numeric, 54 distinct): ['1', '0', '2', '3', '4', '5', '6', '7', '9', '11']
BRANCH_COUNT (numeric, 61 distinct): ['1', '3', '5', '7', '9', '11', '13', '15', '19', '17']
CALL_PAIRS (numeric, 22 distinct): ['1', '0', '2', '3', '4', '5', '6', '8', '7', '9']
LOC_CODE_AND_COMMENT (numeric, 36 distinct): ['0', '1', '2', '3', '4', '6', '5', '9', '7', '8']
LOC_COMMENTS (numeric, 57 distinct): ['0', '1', '2', '3', '6', '4', '5', '9', '10', '7']
CONDITION_COUNT (numeric, 41 distinct): ['0.0', '4.0', '8.0', '12.0', '16.0', '20.0', '6.0', '10.0', '24.0', '22.0']
CYCLOMATIC_COMPLEXITY (numeric, 43 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '10', '9']
CYCLOMATIC_DENSITY (numeric, 70 distinct): ['0.5', '1.0', '0.25', '0.17', '0.33', '0.2', '0.13', '0.14', '0.15', '0.1']
DECISION_COUNT (numeric, 23 distinct): ['0', '2', '4', '6', '8', '10', '12', '14', '24', '18']
DECISION_DENSITY (numeric, 5 distinct): ['0', '2', '3', '4', '5']
DESIGN_COMPLEXITY (numeric, 31 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '13']
DESIGN_DENSITY (numeric, 76 distinct): ['1.0', '0.5', '0.67', '0.33', '0.75', '0.25', '0.8', '0.6', '0.2', '0.57']
EDGE_COUNT (numeric, 105 distinct): ['1.0', '6.0', '2.0', '11.0', '7.0', '5.0', '10.0', '16.0', '9.0', '8.0']
ESSENTIAL_COMPLEXITY (numeric, 25 distinct): ['1', '3', '5', '4', '6', '7', '9', '8', '11', '12']
ESSENTIAL_DENSITY (numeric, 2 distinct): ['0', '1']
LOC_EXECUTABLE (numeric, 107 distinct): ['0', '10', '6', '5', '8', '7', '11', '2', '4', '9']
PARAMETER_COUNT (numeric, 8 distinct): ['0', '1', '2', '3', '4', '5', '7', '6']
HALSTEAD_CONTENT (numeric, 1021 distinct): ['0.0', '18.13', '5.33', '15.86', '9.51', '17.34', '20.0', '8.72', '9.83', '7.75']
HALSTEAD_DIFFICULTY (numeric, 708 distinct): ['0.0', '2.0', '4.0', '3.5', '7.5', '3.0', '1.5', '11.0', '6.0', '8.25']
HALSTEAD_EFFORT (numeric, 1165 distinct): ['0.0', '1020.0', '12.0', '194.27', '139.48', '2087.05', '1180.25', '39.3', '31.02', '152.16']
HALSTEAD_ERROR_EST (numeric, 120 distinct): ['0.0', '0.05', '0.03', '0.01', '0.02', '0.04', '0.07', '0.06', '0.09', '0.08']
HALSTEAD_LENGTH (numeric, 336 distinct): ['0.0', '2.0', '34.0', '48.0', '11.0', '62.0', '27.0', '35.0', '24.0', '33.0']
HALSTEAD_LEVEL (numeric, 40 distinct): ['0.03', '0.05', '0.04', '0.0', '0.02', '0.07', '0.06', '0.1', '0.09', '0.08']
HALSTEAD_PROG_TIME (numeric, 1159 distinct): ['0.0', '56.67', '0.67', '7.75', '10.79', '115.95', '1.72', '65.57', '2.18', '8.45']
HALSTEAD_VOLUME (numeric, 941 distinct): ['0.0', '2.0', '136.0', '34.87', '8.0', '11.61', '77.71', '204.33', '55.51', '18.09']
MAINTENANCE_SEVERITY (numeric, 74 distinct): ['1.0', '0.5', '0.33', '0.25', '0.2', '0.17', '0.14', '0.75', '0.13', '0.6']
MODIFIED_CONDITION_COUNT (numeric, 28 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
MULTIPLE_CONDITION_COUNT (numeric, 40 distinct): ['0', '2', '4', '6', '8', '10', '3', '5', '12', '11']
NODE_COUNT (numeric, 89 distinct): ['2.0', '6.0', '3.0', '10.0', '7.0', '8.0', '9.0', '14.0', '5.0', '12.0']
NORMALIZED_CYLOMATIC_COMPLEXITY (numeric, 67 distinct): ['0.13', '0.08', '0.11', '0.1', '0.2', '0.06', '0.14', '0.25', '0.05', '0.09']
NUM_OPERANDS (numeric, 184 distinct): ['0.0', '13.0', '9.0', '8.0', '17.0', '4.0', '10.0', '6.0', '14.0', '5.0']
NUM_OPERATORS (numeric, 245 distinct): ['0.0', '14.0', '25.0', '2.0', '7.0', '15.0', '9.0', '8.0', '16.0', '17.0']
NUM_UNIQUE_OPERANDS (numeric, 71 distinct): ['6.0', '5.0', '0.0', '7.0', '9.0', '8.0', '3.0', '11.0', '4.0', '12.0']
NUM_UNIQUE_OPERATORS (numeric, 38 distinct): ['11', '10', '8', '7', '13', '12', '9', '6', '14', '15']
NUMBER_OF_LINES (numeric, 171 distinct): ['13.0', '1.0', '14.0', '12.0', '11.0', '8.0', '17.0', '10.0', '7.0', '15.0']
PERCENT_COMMENTS (numeric, 394 distinct): ['0.0', '33.33', '20.0', '25.0', '50.0', '12.5', '14.29', '8.33', '40.0', '11.11']
LOC_TOTAL (numeric, 116 distinct): ['0', '10', '6', '8', '5', '12', '7', '11', '4', '2']
'''

CONTEXT = "Source Code Quality Prediction for PC4"
TARGET = CuratedTarget(raw_name="c", new_name="Is Defective", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []