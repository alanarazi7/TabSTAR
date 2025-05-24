from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Cervical_Cancer_Risk_Factors
====
Examples: 858
====
URL: https://www.openml.org/search?type=data&id=46592
====
Description: Cervical cancer (Risk Factors) Data Set (multilabel classification)

The dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela. The dataset comprises demographic information, habits, and historic medical records of 858 patients. Several patients decided not to answer some of the questions because of privacy concerns (missing values).

Attribute information
(int) Age
(int) Number of sexual partners
(int) First sexual intercourse (age)
(int) Num of pregnancies
(bool) Smokes
(bool) Smokes (years)
(bool) Smokes (packs/year)
(bool) Hormonal Contraceptives
(int) Hormonal Contraceptives (years)
(bool) IUD
(int) IUD (years)
(bool) STDs
(int) STDs (number)
(bool) STDs:condylomatosis
(bool) STDs:cervical condylomatosis
(bool) STDs:vaginal condylomatosis
(bool) STDs:vulvo-perineal condylomatosis
(bool) STDs:syphilis
(bool) STDs:pelvic inflammatory disease
(bool) STDs:genital herpes
(bool) STDs:molluscum contagiosum
(bool) STDs:AIDS
(bool) STDs:HIV
(bool) STDs:Hepatitis B
(bool) STDs:HPV
(int) STDs: Number of diagnosis
(int) STDs: Time since first diagnosis
(int) STDs: Time since last diagnosis
(bool) Dx:Cancer
(bool) Dx:CIN
(bool) Dx:HPV
(bool) Dx
(bool) Hinselmann: target variable
(bool) Schiller: target variable
(bool) Citology: target variable
(bool) Biopsy: target variable

For the current dataset the target variable is a new columnd called "Class", which is the result of the logical OR operation between the columns "Biopsy" "Hinselmann", "Schiller" and "Citology".

{'Biopsy': 1, 'Hinselmann': 2, 'Schiller': 3, 'Citology': 4}, 0 if none of the targets are 1.
====
Target Variable: Class (numeric, 5 distinct): ['0', '1', '4', '3', '2']
====
Features:

Age (numeric, 44 distinct): ['23', '18', '21', '20', '19', '24', '25', '26', '28', '17']
Number of sexual partners (numeric, 12 distinct): ['2.0', '3.0', '1.0', '4.0', '5.0', '6.0', '7.0', '8.0', '15.0', '10.0']
First sexual intercourse (numeric, 21 distinct): ['15.0', '17.0', '18.0', '16.0', '14.0', '19.0', '20.0', '13.0', '21.0', '23.0']
Num of pregnancies (numeric, 11 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '0.0', '7.0', '8.0', '11.0']
Smokes (numeric, 2 distinct): ['0.0', '1.0']
Smokes (years) (numeric, 30 distinct): ['0.0', '1.267', '5.0', '9.0', '1.0', '3.0', '2.0', '16.0', '7.0', '8.0']
Smokes (packs/year) (numeric, 62 distinct): ['0.0', '0.5132', '1.0', '3.0', '2.0', '0.75', '1.2', '0.2', '0.05', '12.0']
Hormonal Contraceptives (numeric, 2 distinct): ['1.0', '0.0']
Hormonal Contraceptives (years) (numeric, 40 distinct): ['0.0', '1.0', '0.25', '2.0', '3.0', '5.0', '0.08', '0.5', '6.0', '4.0']
IUD (numeric, 2 distinct): ['0.0', '1.0']
IUD (years) (numeric, 26 distinct): ['0.0', '3.0', '2.0', '5.0', '1.0', '8.0', '7.0', '6.0', '4.0', '11.0']
STDs (numeric, 2 distinct): ['0.0', '1.0']
STDs (number) (numeric, 5 distinct): ['0.0', '2.0', '1.0', '3.0', '4.0']
STDs:condylomatosis (numeric, 2 distinct): ['0.0', '1.0']
STDs:cervical condylomatosis (numeric, 1 distinct): ['0.0']
STDs:vaginal condylomatosis (numeric, 2 distinct): ['0.0', '1.0']
STDs:vulvo-perineal condylomatosis (numeric, 2 distinct): ['0.0', '1.0']
STDs:syphilis (numeric, 2 distinct): ['0.0', '1.0']
STDs:pelvic inflammatory disease (numeric, 2 distinct): ['0.0', '1.0']
STDs:genital herpes (numeric, 2 distinct): ['0.0', '1.0']
STDs:molluscum contagiosum (numeric, 2 distinct): ['0.0', '1.0']
STDs:AIDS (numeric, 1 distinct): ['0.0']
STDs:HIV (numeric, 2 distinct): ['0.0', '1.0']
STDs:Hepatitis B (numeric, 2 distinct): ['0.0', '1.0']
STDs:HPV (numeric, 2 distinct): ['0.0', '1.0']
STDs: Number of diagnosis (numeric, 4 distinct): ['0', '1', '2', '3']
STDs: Time since first diagnosis (numeric, 18 distinct): ['1.0', '3.0', '2.0', '4.0', '7.0', '5.0', '16.0', '6.0', '8.0', '21.0']
STDs: Time since last diagnosis (numeric, 18 distinct): ['1.0', '2.0', '3.0', '4.0', '7.0', '16.0', '5.0', '6.0', '8.0', '21.0']
Dx:Cancer (numeric, 2 distinct): ['0', '1']
Dx:CIN (numeric, 2 distinct): ['0', '1']
Dx:HPV (numeric, 2 distinct): ['0', '1']
Dx (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Cervical Cancer Risk Factors in Venezuela"
TARGET = CuratedTarget(raw_name="Class", new_name="Cervical Cancer Risk", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"0": "No Cancer", "1": "Biopsy", "2": "Hinselmann",
                                      "3": "Schiller", "4": "Citology"})
COLS_TO_DROP = []
FEATURES = []