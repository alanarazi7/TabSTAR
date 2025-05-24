from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Diabetes130US
====
Examples: 101766
====
URL: https://www.openml.org/search?type=data&id=4541
====
Description: **Author**: Attila Reiss, Department Augmented Vision, DFKI, Germany, "attila.reiss '@' dfki.de  
**Date**: August 2012.  
**Source**: UCI  
**Please cite**: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, &ldquo; Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,&rdquo; BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.  

This data has been prepared to analyze factors related to readmission as well as other outcomes pertaining to patients with diabetes.

**Source**  
The data are submitted on behalf of the Center for Clinical and Translational Research, Virginia Commonwealth University, a recipient of NIH CTSA grant UL1 TR00058 and a recipient of the CERNER data. John Clore (jclore '@' vcu.edu), Krzysztof J. Cios (kcios '@' vcu.edu), Jon DeShazo (jpdeshazo '@' vcu.edu), and Beata Strack (strackb '@' vcu.edu). This data is a de-identified abstract of the Health Facts database (Cerner Corporation, Kansas City, MO).

**Data Set Information**  
The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria:  
(1) It is an inpatient encounter (a hospital admission).  
(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.  
(3) The length of stay was at least 1 day and at most 14 days.  
(4) Laboratory tests were performed during the encounter.  
(5) Medications were administered during the encounter.  
The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

**Attribute Information**  
Detailed description of all the attributes is provided in Table 1 of the paper.  

**Relevant Papers**  
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, &ldquo;Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,&rdquo; BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

[Web Link](https://www.hindawi.com/journals/bmri/2014/781670/)

Encounter ID	Numeric	Unique identifier of an encounter	0%
Patient number	Numeric	Unique identifier of a patient	0%
Race	Nominal	Values: Caucasian, Asian, African American, Hispanic, and other	2%
Gender	Nominal	Values: male, female, and unknown/invalid	0%
Age	Nominal	Grouped in 10-year intervals: [0, 10), [10, 20), …, [90, 100)	0%
Weight	Numeric	Weight in pounds.	97%
Admission type	Nominal	Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available	0%
Discharge disposition	Nominal	Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available	0%
Admission source	Nominal	Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital	0%
Time in hospital	Numeric	Integer number of days between admission and discharge	0%
Payer code	Nominal	Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay	52%
Medical specialty	Nominal	Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon	53%
Number of lab procedures	Numeric	Number of lab tests performed during the encounter	0%
Number of procedures	Numeric	Number of procedures (other than lab tests) performed during the encounter	0%
Number of medications	Numeric	Number of distinct generic names administered during the encounter	0%
Number of outpatient visits	Numeric	Number of outpatient visits of the patient in the year preceding the encounter	0%
Number of emergency visits	Numeric	Number of emergency visits of the patient in the year preceding the encounter	0%
Number of inpatient visits	Numeric	Number of inpatient visits of the patient in the year preceding the encounter	0%
Diagnosis 1	Nominal	The primary diagnosis (coded as first three digits of ICD9); 848 distinct values	0%
Diagnosis 2	Nominal	Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values	0%
Diagnosis 3	Nominal	Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values	1%
Number of diagnoses	Numeric	Number of diagnoses entered to the system	0%
Glucose serum test result	Nominal	Indicates the range of the result or if the test was not taken. Values: “>200,” “>300,” “normal,” and “none” if not measured	0%
A1c test result	Nominal	Indicates the range of the result or if the test was not taken. Values: “>8” if the result was greater than 8%, “>7” if the result was greater than 7% but less than 8%, “normal” if the result was less than 7%, and “none” if not measured.	0%
Change of medications	Nominal	Indicates if there was a change in diabetic medications (either dosage or generic name). Values: “change” and “no change”	0%
Diabetes medications	Nominal	Indicates if there was any diabetic medication prescribed. Values: “yes” and “no”	0%
24 features for medications	Nominal	For the generic names: metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, sitagliptin, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, and metformin-pioglitazone, the feature indicates whether the drug was prescribed or there was a change in the dosage. Values: “up” if the dosage was increased during the encounter, “down” if the dosage was decreased, “steady” if the dosage did not change, and “no” if the drug was not prescribed	0%
Readmitted	Nominal	Days to inpatient readmission. Values: “<30” if the patient was readmitted in less than 30 days, “>30” if the patient was readmitted in more than 30 days, and “No” for no record of readmission.
====
Target Variable: readmitted (nominal, 3 distinct): ['NO', '>30', '<30']
====
Features:

encounter_id (numeric, 101766 distinct): ['2278392.0', '190792044.0', '190790070.0', '190789722.0', '190786806.0', '190785018.0', '190781412.0', '190775886.0', '190764504.0', '190760322.0']
patient_nbr (numeric, 71518 distinct): ['88785891.0', '43140906.0', '1660293.0', '88227540.0', '23199021.0', '23643405.0', '84428613.0', '92709351.0', '88789707.0', '29903877.0']
race (nominal, 6 distinct): ['Caucasian', 'AfricanAmerican', '?', 'Hispanic', 'Other', 'Asian']
gender (nominal, 3 distinct): ['Female', 'Male', 'Unknown/Invalid']
age (nominal, 10 distinct): ['[70-80)', '[60-70)', '[50-60)', '[80-90)', '[40-50)', '[30-40)', '[90-100)', '[20-30)', '[10-20)', '[0-10)']
weight (nominal, 10 distinct): ['?', '[75-100)', '[50-75)', '[100-125)', '[125-150)', '[25-50)', '[0-25)', '[150-175)', '[175-200)', '>200']
admission_type_id (numeric, 8 distinct): ['1', '3', '2', '6', '5', '8', '7', '4']
discharge_disposition_id (numeric, 26 distinct): ['1', '3', '6', '18', '2', '22', '11', '5', '25', '4']
admission_source_id (numeric, 17 distinct): ['7', '1', '17', '4', '6', '2', '5', '3', '20', '9']
time_in_hospital (numeric, 14 distinct): ['3', '2', '1', '4', '5', '6', '7', '8', '9', '10']
payer_code (nominal, 18 distinct): ['?', 'MC', 'HM', 'SP', 'BC', 'MD', 'CP', 'UN', 'CM', 'OG']
medical_specialty (nominal, 73 distinct): ['?', 'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Radiologist']
num_lab_procedures (numeric, 118 distinct): ['1', '43', '44', '45', '38', '40', '46', '41', '42', '47']
num_procedures (numeric, 7 distinct): ['0', '1', '2', '3', '6', '4', '5']
num_medications (numeric, 75 distinct): ['13', '12', '11', '15', '14', '16', '10', '17', '9', '18']
number_outpatient (numeric, 39 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
number_emergency (numeric, 33 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
number_inpatient (numeric, 21 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
diag_1 (nominal, 717 distinct): ['428', '414', '786', '410', '486', '427', '491', '715', '682', '434']
diag_2 (nominal, 749 distinct): ['276', '428', '250', '427', '401', '496', '599', '403', '414', '411']
diag_3 (nominal, 790 distinct): ['250', '401', '276', '428', '427', '414', '496', '403', '585', '272']
number_diagnoses (numeric, 16 distinct): ['9', '5', '8', '7', '6', '4', '3', '2', '1', '16']
max_glu_serum (nominal, 4 distinct): ['None', 'Norm', '>200', '>300']
A1Cresult (nominal, 4 distinct): ['None', '>8', 'Norm', '>7']
metformin (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
repaglinide (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
nateglinide (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
chlorpropamide (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
glimepiride (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
acetohexamide (nominal, 2 distinct): ['No', 'Steady']
glipizide (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
glyburide (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
tolbutamide (nominal, 2 distinct): ['No', 'Steady']
pioglitazone (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
rosiglitazone (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
acarbose (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
miglitol (nominal, 4 distinct): ['No', 'Steady', 'Down', 'Up']
troglitazone (nominal, 2 distinct): ['No', 'Steady']
tolazamide (nominal, 3 distinct): ['No', 'Steady', 'Up']
examide (nominal, 1 distinct): ['No']
citoglipton (nominal, 1 distinct): ['No']
insulin (nominal, 4 distinct): ['No', 'Steady', 'Down', 'Up']
glyburide.metformin (nominal, 4 distinct): ['No', 'Steady', 'Up', 'Down']
glipizide.metformin (nominal, 2 distinct): ['No', 'Steady']
glimepiride.pioglitazone (nominal, 2 distinct): ['No', 'Steady']
metformin.rosiglitazone (nominal, 2 distinct): ['No', 'Steady']
metformin.pioglitazone (nominal, 2 distinct): ['No', 'Steady']
change (nominal, 2 distinct): ['No', 'Ch']
diabetesMed (nominal, 2 distinct): ['Yes', 'No']
'''

CONTEXT = "Diabetes Patients Readmission Prediction in US Hospitals"
TARGET = CuratedTarget(raw_name="readmitted", new_name="Patient Readmission", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'NO': 'No Readmission',
                                      '>30': 'Readmitted in more than 30 days',
                                      '<30': 'Readmitted in less than 30 days'})
COLS_TO_DROP = ["encounter_id"]
FEATURES = []