from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: BAF_base
====
Examples: 1000000
====
URL: https://www.openml.org/search?type=data&id=46646
====
Description: Evaluating new techniques on realistic datasets plays a crucial role in the development of ML research and its broader adoption by practitioners. In recent years,
there has been a significant increase of publicly available unstructured data resources for computer vision and NLP tasks. However, tabular data - which is
prevalent in many high-stakes domains - has been lagging behind. To bridge this
gap, we present Bank Account Fraud (BAF), the first publicly available1 privacypreserving, large-scale, realistic suite of tabular datasets. The suite was generated
by applying state-of-the-art tabular data generation techniques on an anonymized,
real-world bank account opening fraud detection dataset. This setting carries a set
of challenges that are commonplace in real-world applications, including temporal
dynamics and significant class imbalance. Additionally, to allow practitioners to
stress test both performance and fairness of ML methods, each dataset variant of
BAF contains specific types of data bias. With this resource, we aim to provide the
research community with a more realistic, complete, and robust test bed to evaluate
novel and existing method
====
Target Variable: fraud_bool (numeric, 2 distinct): ['0', '1']
====
Features:

income (numeric, 9 distinct): ['0.9', '0.1', '0.8', '0.6', '0.7', '0.4', '0.2', '0.5', '0.3']
name_email_similarity (numeric, 998861 distinct): ['0.3893', '0.5331', '0.6703', '0.0127', '0.6218', '0.828', '0.2955', '0.7874', '0.745', '0.522']
prev_address_months_count (numeric, 374 distinct): ['-1', '11', '28', '29', '10', '27', '30', '26', '12', '31']
current_address_months_count (numeric, 423 distinct): ['6', '7', '8', '5', '9', '4', '10', '11', '3', '12']
customer_age (numeric, 9 distinct): ['30', '20', '40', '50', '60', '10', '70', '80', '90']
days_since_request (numeric, 989330 distinct): ['0.014', '0.0285', '0.0285', '0.0188', '0.0228', '0.0074', '0.0158', '0.0337', '0.0258', '0.0186']
intended_balcon_amount (numeric, 994971 distinct): ['-0.469', '-0.766', '-0.7027', '-1.1891', '-0.8829', '-0.5041', '-1.543', '-1.5321', '-0.7245', '-1.5143']
payment_type (string, 5 distinct): ['AB', 'AA', 'AC', 'AD', 'AE']
zip_count_4w (numeric, 6306 distinct): ['1020', '1062', '1042', '969', '1026', '996', '1030', '1033', '1102', '1112']
velocity_6h (numeric, 998687 distinct): ['8506.534', '4181.3494', '4407.3001', '3638.5187', '4366.0493', '3050.2719', '7639.524', '3163.5801', '8996.1857', '4793.3287']
velocity_24h (numeric, 998940 distinct): ['6676.2013', '5082.3266', '3906.4163', '4667.8639', '5768.9046', '4001.2535', '6171.5102', '3998.9103', '4736.6158', '5581.5228']
velocity_4w (numeric, 998318 distinct): ['5466.8709', '4297.6649', '5599.84', '5604.141', '4923.4233', '4370.3195', '4315.903', '5478.5574', '4329.1495', '5449.3839']
bank_branch_count_8w (numeric, 2326 distinct): ['1', '0', '2', '11', '10', '12', '9', '13', '8', '14']
date_of_birth_distinct_emails_4w (numeric, 40 distinct): ['7', '5', '8', '6', '11', '9', '10', '4', '13', '12']
employment_status (string, 7 distinct): ['CA', 'CB', 'CF', 'CC', 'CD', 'CE', 'CG']
credit_risk_score (numeric, 551 distinct): ['113', '116', '110', '115', '117', '114', '109', '105', '112', '111']
email_is_free (numeric, 2 distinct): ['1', '0']
housing_status (string, 7 distinct): ['BC', 'BB', 'BA', 'BE', 'BD', 'BF', 'BG']
phone_home_valid (numeric, 2 distinct): ['0', '1']
phone_mobile_valid (numeric, 2 distinct): ['1', '0']
bank_months_count (numeric, 33 distinct): ['-1', '1', '28', '15', '30', '31', '25', '10', '20', '21']
has_other_cards (numeric, 2 distinct): ['0', '1']
proposed_credit_limit (numeric, 12 distinct): ['200.0', '1500.0', '500.0', '1000.0', '990.0', '510.0', '2000.0', '490.0', '210.0', '1900.0']
foreign_request (numeric, 2 distinct): ['0', '1']
source (string, 2 distinct): ['INTERNET', 'TELEAPP']
session_length_in_minutes (numeric, 994887 distinct): ['-1.0', '9.662', '7.9601', '4.1924', '2.4637', '1.8442', '5.1016', '4.4439', '4.6387', '13.1526']
device_os (string, 5 distinct): ['other', 'linux', 'windows', 'macintosh', 'x11']
keep_alive_session (numeric, 2 distinct): ['1', '0']
device_distinct_emails_8w (numeric, 4 distinct): ['1', '2', '0', '-1']
device_fraud_count (numeric, 1 distinct): ['0']
month (numeric, 8 distinct): ['3', '2', '0', '4', '1', '5', '6', '7']
'''

CONTEXT = "Bank Account Fraud (BAF)"
TARGET = CuratedTarget(raw_name="fraud_bool", new_name="Fraud", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []