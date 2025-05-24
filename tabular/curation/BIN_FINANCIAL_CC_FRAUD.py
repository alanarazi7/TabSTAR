from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Credit_Card_Fraud_
====
Examples: 1000000
====
URL: https://www.openml.org/search?type=data&id=45955
====
Description: **Dataset Name**: card_transdata.csv

**Description**:
This dataset captures transaction patterns and behaviors that could indicate potential fraud in card transactions. The data is composed of several features designed to reflect the transactional context such as geographical location, transaction medium, and spending behavior relative to the user's history.

**Attribute Description**:
1. **distance_from_home**: This is a numerical feature representing the geographical distance in kilometers between the transaction location and the cardholder's home address.
2. **distance_from_last_transaction**: This numerical attribute measures the distance in kilometers from the location of the last transaction to the current transaction location.
3. **ratio_to_median_purchase_price**: A numeric ratio that compares the transaction's price to the median purchase price of the user's transaction history.
4. **repeat_retailer**: A binary attribute where '1' signifies that the transaction was conducted at a retailer previously used by the cardholder, and '0' indicates a new retailer.
5. **used_chip**: This binary feature indicates whether the transaction was made using a chip (1) or not (0).
6. **used_pin_number**: Another binary feature, where '1' signifies the use of a PIN number for the transaction, and '0' shows no PIN number was used.
7. **online_order**: This attribute identifies whether the purchase was made online ('1') or offline ('0').
8. **fraud**: A binary target variable indicating whether the transaction was fraudulent ('1') or not ('0').

**Use Case**:
This dataset is particularly suited for developing machine learning models to detect potentially fraudulent transactions. Financial institutions and cybersecurity firms can leverage this data to enhance their fraud detection systems, ensuring safer transaction environments for cardholders. Researchers in fintech and cybersecurity can also use this dataset for academic purposes, exploring new methodologies in fraud detection algorithms. Additionally, policy makers and regulatory bodies can analyze trends and patterns to formulate guidelines that mitigate transactional fraud.
====
Target Variable: fraud (numeric, 2 distinct): ['0', '1']
====
Features:

distance_from_home (numeric, 999999 distinct): ['29.5292', '17.6967', '47.9332', '48.3007', '25.4144', '0.3378', '0.7204', '18.8369', '2.4209', '36.572']
distance_from_last_transaction (numeric, 999956 distinct): ['0.0287', '0.0533', '0.13', '0.1234', '0.0716', '0.0974', '0.0885', '0.0795', '2.6728', '0.077']
ratio_to_median_purchase_price (numeric, 999974 distinct): ['0.4552', '0.3483', '0.3898', '1.1082', '0.1866', '1.2037', '0.6181', '1.221', '0.3176', '0.3913']
repeat_retailer (numeric, 2 distinct): ['1', '0']
used_chip (numeric, 2 distinct): ['0', '1']
used_pin_number (numeric, 2 distinct): ['0', '1']
online_order (numeric, 2 distinct): ['1', '0']
'''

CONTEXT = "Credit Card Transactions for Fraud Detection"
TARGET = CuratedTarget(raw_name="fraud", new_name="Credit Card Fraudulent Transaction", task_type=SupervisedTask.BINARY,
                       label_mapping={"0": "Not Fraudulent", "1": "Fraudulent"})
COLS_TO_DROP = []
FEATURES = []
