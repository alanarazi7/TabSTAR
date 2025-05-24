from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Breast-cancer-prediction
====
Examples: 39998
====
URL: https://www.openml.org/search?type=data&id=43687
====
Description: Context
This dataset includes data from a random sample of 20,000 digital and 20,000 film-screen mammograms received by women age 60-89 years within the Breast Cancer Surveillance Consortium (BCSC) between January 2005 and December 2008. Some women contribute multiple examinations to the dataset. Data is useful in teaching about data analysis, epidemiological study designs, or statistical methods for binary outcomes or correlated data.
Content
The data set contains 39998 rows and  13 cols. Attributes are described as follows:
  Field Name     **Type (Format) **Description



AgeAtTheTimeOf_Mammography
number
Patient's age in years at time of mammogram




Radiologists_Assessment
string
Radiologist's assessment based on the BI-RADS scale


---
---
---


IsBinaryIndicatorOfCancer_Diagnosis
boolean
Binary indicator of cancer diagnosis within one year of screening mammogram (false= No cancer diagnosis, true= Cancer diagnosis)


---
---
---


ComparisonMammogramFrom_Mammography
string
Comparison mammogram from prior mammography examination available


---
---
---


PatientsBIRADSBreastDensity
string
Patient's BI-RADS breast density as recorded at time of mammogram


---
---
---


FamilyHistoryOfBreastCancer
string
Family history of breast cancer in a first degree relative


---
---
---


CurrentUseOfHormoneTherapy
string
Current use of hormone therapy at time of mammogram


---
---
---


Binary_Indicator
string
Binary indicator of whether the woman had ever received a prior mammogram


---
---
---


HistoryOfBreast_Biopsy
string
Prior history of breast biopsy


---
---
---


IsFilmOrDigitalMammogram
boolean
Film or digital mammogram (true=Digital mammogram, false=Film mammogram)


---
---
---


Cancer_Type
string
Type of cancer


---
---
---



Acknowledgements
We acknowledge the Breast Cancer Surveillance Consortium (BCSC) for making this data set available for research purposes.
====
Target Variable: Is_Binary_Indicator_Of_Cancer_Diagnosis (nominal, 2 distinct): ['0', '1']
====
Features:

Age_At_The_Time_Of_Mammography (numeric, 30 distinct): ['60', '61', '62', '63', '65', '64', '66', '67', '68', '69']
Radiologists_Assessment (string, 6 distinct): ['Negative', 'Benign findings', 'Needs additional imaging', 'Probably benign', 'Suspicious abnormality', 'Highly suggestive of malignancy']
Comparison_Mammogram_From_Mammography (string, 3 distinct): ['Yes', 'Missing', 'No']
Patients_BI_RADS_Breast_Density (string, 4 distinct): ['Scattered fibroglandular densities', 'Heterogeneously dense', 'Almost entirely fatty', 'Extremely dense']
Family_History_Of_Breast_Cancer (string, 3 distinct): ['No', 'Yes', 'Missing']
Current_Use_Of_Hormone_Therapy (string, 3 distinct): ['No', 'Yes', 'Missing']
Binary_Indicator (string, 3 distinct): ['Yes', 'Missing', 'No']
History_Of_Breast_Biopsy (string, 3 distinct): ['No', 'Yes', 'Missing']
Is_Film_Or_Digital_Mammogram (nominal, 2 distinct): ['1', '0']
Cancer_Type (string, 3 distinct): ['No cancer diagnosis', 'Invasive cancer', 'ductal carcinoma in situ']
Body_Mass_Index (string, 1897 distinct): ['Missing', '25.6060486', '24.7996063', '27.463623', '29.2640533', '25.7471466', '23.7770386', '28.3424072', '24.9610291', '27.4350433']
'''

CONTEXT = "Mammograms by woman within the Breast Cancer Surveillance Consortium"
TARGET = CuratedTarget(raw_name="Is_Binary_Indicator_Of_Cancer_Diagnosis", new_name="Cancer Diagnosis",
                       task_type=SupervisedTask.BINARY, label_mapping={'False': "No Cancer", 'True': "Cancer"})
COLS_TO_DROP = []
FEATURES = []
