from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: differentiated_thyroid_cancer_recurrence
====
Examples: 383
====
URL: https://www.openml.org/search?type=data&id=46605
====
Description: Description: The dataset is a comprehensive collection of clinical data relating to thyroid diseases. With attributes capturing a wide range of information from patient demographics (age, gender) to specific clinical findings (smoking history, radiotherapy history, thyroid function, physical examination findings), it provides a detailed overview of patients diagnosed with various forms of thyroid conditions. The dataset encapsulates aspects such as the presence of adenopathy, pathology findings, focality of the disease, and risk categorization. Further, it delves into the TNM classification system, providing insights into the size and extent of tumors (T), presence of cancer in nearby lymph nodes (N), and metastasis (M), thereby contributing to the staging of the disease. The clinical response to treatment and recurrence status is also recorded, offering valuable data for outcomes analysis.

Attribute Description:

Age: Numeric, represents the age of the patient.
Gender: Categorical, 'M' for male, 'F' for female.
Smoking: Binary, 'Yes' if the patient has a history of smoking, 'No' otherwise.
Hx Smoking: Binary, indicating a historical record of smoking.
Hx Radiotherapy: Binary, indicates if the patient has undergone radiotherapy.
Thyroid Function: Categorical, reports the thyroid's functional state.
Physical Examination: Text, describes findings from physical examination.
Adenopathy: Binary, 'Yes' if adenopathy is present, 'No' otherwise.
Pathology: Categorical, type of thyroid pathology diagnosed.
Focality: Categorical, 'Multi-Focal' or 'Uni-Focal' disease spread.
Risk: Categorical, assessed risk level ('Low', 'Intermediate', 'High').
T, N, M: Staging parameters as per the TNM classification.
Stage: Categorical, stage of the disease.
Response: Categorical, patient's response to treatment.
Recurred: Binary, 'Yes' if the disease has recurred, 'No' otherwise.
Use Case: This dataset is instrumental for researchers and clinicians focusing on thyroid diseases. Its detailed attributes facilitate analyses on the relationship between demographic factors, lifestyle choices (such as smoking), clinical findings, and treatment outcomes. Furthermore, it can serve as a valuable resource for predictive modeling of disease progression, recurrence, and response to therapy. Machine learning applications can leverage this dataset for developing algorithms that predict patient outcomes, guide treatment plans, and assess risk factors for disease recurrence or poor treatment response.

For what purpose was the dataset created?

It was a part of research in the field of AI and Medicine

Who funded the creation of the dataset?

No funding was provided.

What do the instances in this dataset represent?

Individual patients

Are there recommended data splits?

No

Does the dataset contain data that might be considered sensitive in any way?

No

Has Missing Values?

No
====
Target Variable: Recurred (string, 2 distinct): ['No', 'Yes']
====
Features:

Age (numeric, 65 distinct): ['31', '27', '40', '26', '28', '35', '30', '33', '34', '29']
Gender (string, 2 distinct): ['F', 'M']
Smoking (string, 2 distinct): ['No', 'Yes']
Hx Smoking (string, 2 distinct): ['No', 'Yes']
Hx Radiothreapy (string, 2 distinct): ['No', 'Yes']
Thyroid Function (string, 5 distinct): ['Euthyroid', 'Clinical Hyperthyroidism', 'Subclinical Hypothyroidism', 'Clinical Hypothyroidism', 'Subclinical Hyperthyroidism']
Physical Examination (string, 5 distinct): ['Multinodular goiter', 'Single nodular goiter-right', 'Single nodular goiter-left', 'Normal', 'Diffuse goiter']
Adenopathy (string, 6 distinct): ['No', 'Right', 'Bilateral', 'Left', 'Extensive', 'Posterior']
Pathology (string, 4 distinct): ['Papillary', 'Micropapillary', 'Follicular', 'Hurthel cell']
Focality (string, 2 distinct): ['Uni-Focal', 'Multi-Focal']
Risk (string, 3 distinct): ['Low', 'Intermediate', 'High']
T (string, 7 distinct): ['T2', 'T3a', 'T1a', 'T1b', 'T4a', 'T3b', 'T4b']
N (string, 3 distinct): ['N0', 'N1b', 'N1a']
M (string, 2 distinct): ['M0', 'M1']
Stage (string, 5 distinct): ['I', 'II', 'IVB', 'III', 'IVA']
Response (string, 4 distinct): ['Excellent', 'Structural Incomplete', 'Indeterminate', 'Biochemical Incomplete']
'''

CONTEXT = "Thyroid Cancer Recurrence"
TARGET = CuratedTarget(raw_name="Recurred", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []