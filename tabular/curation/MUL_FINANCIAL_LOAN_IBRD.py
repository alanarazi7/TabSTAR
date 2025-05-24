from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: IBRD_Loans_Classification
====
Examples: 9217
====
URL: https://www.openml.org/search?type=data&id=46467
====
Description: The International Bank for Reconstruction and Development (IBRD) loans are public and publicly guaranteed debt extended by the World Bank Group. IBRD loans are made to, or guaranteed by, countries/economies that are members of IBRD. IBRD may also make loans to IFC. IBRD lends at market rates. Data are in U.S. dollars calculated using historical rates. This dataset contains the latest available snapshot of the Statement of Loans. The World Bank complies with all sanctions applicable to World Bank transactions. 

The dataset provides insights into loan types, statuses, financial amounts, and other relevant details. It is useful for classification tasks focused on predicting loan types.
====
Target Variable: Loan_Type (nominal, 11 distinct): ['FSL', 'CPL', 'NPL', 'SCL', 'SCPD', 'SCPM', 'GURB', 'BLNR', 'SCPY', 'GUBF']
====
Features:

Loan_Number (string, 9217 distinct): ['IBRDB0050', 'IBRD75220', 'IBRD09907', 'IBRD09910', 'IBRD09920', 'IBRD09930', 'IBRD09940', 'IBRD09950', 'IBRD09960', 'IBRD09970']
Loan_Status (string, 11 distinct): ['Fully Repaid', 'Repaying', 'Disbursing', 'Fully Disbursed', 'Fully Cancelled', 'Fully Transferred', 'Disbursing&Repaying', 'Terminated', 'Approved', 'Signed']
Interest_Rate (numeric, 461 distinct): ['0.0', '7.25', '8.5', '6.03', '5.59', '7.43', '11.6', '7.9', '5.5', '8.25']
Original_Principal_Amount_(US$) (numeric, 3453 distinct): ['100000000.0', '0.0', '50000000.0', '200000000.0', '150000000.0', '30000000.0', '20000000.0', '300000000.0', '25000000.0', '40000000.0']
Cancelled_Amount_(US$) (numeric, 4395 distinct): ['0.0', '20000000.0', '100000000.0', '50000000.0', '5000000.0', '10000000.0', '30000000.0', '25000000.0', '40000000.0', '200000000.0']
Disbursed_Amount_(US$) (numeric, 6772 distinct): ['0.0', '100000000.0', '25000000.0', '50000000.0', '150000000.0', '200000000.0', '300000000.0', '20000000.0', '500000000.0', '30000000.0']
Repaid_to_IBRD_(US$) (numeric, 6837 distinct): ['0.0', '100000000.0', '30000000.0', '50000000.0', '25000000.0', '150000000.0', '20000000.0', '300000000.0', '10000000.0', '200000000.0']
'''

CONTEXT = "International Bank for Reconstruction and Development (IBRD) Loans Classification"
TARGET = CuratedTarget(raw_name="Loan_Type", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["Loan_Number"]
FEATURES = []

