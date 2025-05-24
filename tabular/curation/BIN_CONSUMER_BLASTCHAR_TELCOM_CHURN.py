from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: blastchar
====
Examples: 7043
====
URL: https://www.openml.org/search?type=data&id=46280
====
Description: From original source:
-----

Context
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

Content
Each row represents a customer, each column contains customer's attributes described on the column Metadata.

The data set includes information about:

Customers who left within the last month - the column is called Churn
Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers - gender, age range, and if they have partners and dependents
Inspiration
To explore this type of models and learn more about the subject.

New version from IBM:
https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
-----
Columns with index 0 were deleted from the dataset, usually because they related to some kind of index.
====
Target Variable: Churn (nominal, 2 distinct): ['No', 'Yes']
====
Features:

gender (nominal, 2 distinct): ['Male', 'Female']
SeniorCitizen (nominal, 2 distinct): ['0', '1']
Partner (nominal, 2 distinct): ['No', 'Yes']
Dependents (nominal, 2 distinct): ['No', 'Yes']
tenure (numeric, 73 distinct): ['1', '72', '2', '3', '4', '71', '5', '7', '8', '70']
PhoneService (nominal, 2 distinct): ['Yes', 'No']
MultipleLines (nominal, 3 distinct): ['No', 'Yes', 'No phone service']
InternetService (nominal, 3 distinct): ['Fiber optic', 'DSL', 'No']
OnlineSecurity (nominal, 3 distinct): ['No', 'Yes', 'No internet service']
OnlineBackup (nominal, 3 distinct): ['No', 'Yes', 'No internet service']
DeviceProtection (nominal, 3 distinct): ['No', 'Yes', 'No internet service']
TechSupport (nominal, 3 distinct): ['No', 'Yes', 'No internet service']
StreamingTV (nominal, 3 distinct): ['No', 'Yes', 'No internet service']
StreamingMovies (nominal, 3 distinct): ['No', 'Yes', 'No internet service']
Contract (nominal, 3 distinct): ['Month-to-month', 'Two year', 'One year']
PaperlessBilling (nominal, 2 distinct): ['Yes', 'No']
PaymentMethod (nominal, 4 distinct): ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
MonthlyCharges (numeric, 1585 distinct): ['20.05', '19.85', '19.95', '19.9', '20.0', '19.7', '19.65', '19.55', '20.15', '19.75']
TotalCharges (numeric, 6531 distinct): ['20.2', '19.75', '20.05', '19.9', '19.65', '19.55', '45.3', '19.45', '20.25', '20.15']
'''

CONTEXT = "Telcom Company Customer Behavior for Churn Prediction"
TARGET = CuratedTarget(raw_name="Churn", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []