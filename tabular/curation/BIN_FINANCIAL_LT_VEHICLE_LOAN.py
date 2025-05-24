from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: LT-Vehicle-Loan-Default-Prediction
====
Examples: 233154
====
URL: https://www.openml.org/search?type=data&id=46430
====
Description: Dataset is uploaded from kaggle, see citation for the link.

https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction

Financial institutions incur significant losses due to the default of vehicle loans. This has led to the tightening up of vehicle loan underwriting and increased vehicle loan rejection rates. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default. A financial institution has hired you to accurately predict the probability of loanee/borrower defaulting on a vehicle loan in the first EMI (Equated Monthly Instalments) on the due date. Following Information regarding the loan and loanee are provided in the datasets:
Loanee Information (Demographic data like age, Identity proof etc.)
Loan Information (Disbursal details, loan to value ratio etc.)
Bureau data & history (Bureau score, number of active accounts, the status of other loans, credit history etc.)
Doing so will ensure that clients capable of repayment are not rejected and important determinants can be identified which can be further used for minimising the default rates.


====
Target Variable: loan_default (nominal, 2 distinct): ['0', '1']
====
Features:

uniqueid (numeric, 233154 distinct): ['420825', '573390', '443579', '634411', '497340', '613162', '651043', '443927', '430582', '437246']
disbursed_amount (numeric, 24565 distinct): ['48349', '53303', '51303', '50303', '55259', '52303', '47349', '56259', '46349', '57259']
asset_cost (numeric, 46252 distinct): ['68000', '67000', '72000', '70000', '74000', '66000', '73000', '75000', '69000', '65000']
ltv (numeric, 6579 distinct): ['85.0', '84.99', '79.99', '80.0', '75.0', '79.9', '79.79', '74.93', '90.0', '74.99']
branch_id (numeric, 82 distinct): ['2', '67', '3', '5', '36', '136', '34', '16', '19', '1']
supplier_id (numeric, 2953 distinct): ['18317', '15694', '15663', '17980', '14234', '18166', '21980', '14375', '22727', '14145']
manufacturer_id (numeric, 11 distinct): ['86', '45', '51', '48', '49', '120', '67', '145', '153', '152']
current_pincode_id (numeric, 6698 distinct): ['2578', '1446', '1515', '2989', '2943', '1509', '2782', '1794', '571', '3363']
date.of.birth (string, 15433 distinct): ['01-01-88', '01-01-90', '01-01-87', '01-01-86', '01-01-85', '01-01-91', '01-01-89', '01-01-93', '01-01-95', '01-01-92']
employment.type (string, 3 distinct): ['Self employed', 'Salaried']
disbursaldate (string, 84 distinct): ['31-10-18', '24-10-18', '31-08-18', '23-10-18', '26-10-18', '25-10-18', '22-10-18', '30-10-18', '30-08-18', '29-10-18']
state_id (numeric, 22 distinct): ['4', '3', '6', '13', '9', '8', '5', '14', '1', '7']
employee_code_id (numeric, 3270 distinct): ['2546', '620', '255', '130', '2153', '956', '184', '1466', '1494', '64']
mobileno_avl_flag (numeric, 1 distinct): ['1']
aadhar_flag (numeric, 2 distinct): ['1', '0']
pan_flag (numeric, 2 distinct): ['0', '1']
voterid_flag (numeric, 2 distinct): ['0', '1']
driving_flag (numeric, 2 distinct): ['0', '1']
passport_flag (numeric, 2 distinct): ['0', '1']
perform_cns.score (numeric, 573 distinct): ['0', '300', '738', '825', '15', '17', '763', '16', '708', '737']
perform_cns.score.description (string, 20 distinct): ['No Bureau History Available', 'C-Very Low Risk', 'A-Very Low Risk', 'D-Very Low Risk', 'B-Very Low Risk', 'M-Very High Risk', 'F-Low Risk', 'K-High Risk', 'H-Medium Risk', 'E-Low Risk']
pri.no.of.accts (numeric, 108 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pri.active.accts (numeric, 40 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pri.overdue.accts (numeric, 22 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pri.current.balance (numeric, 71341 distinct): ['0', '800', '400', '30000', '50000', '100000', '40000', '25000', '20000', '60000']
pri.sanctioned.amount (numeric, 44390 distinct): ['0', '50000', '30000', '100000', '25000', '40000', '20000', '60000', '200000', '15000']
pri.disbursed.amount (numeric, 47909 distinct): ['0', '50000', '30000', '100000', '40000', '25000', '20000', '200000', '300000', '60000']
sec.no.of.accts (numeric, 37 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sec.active.accts (numeric, 23 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sec.overdue.accts (numeric, 9 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7']
sec.current.balance (numeric, 3246 distinct): ['0', '800', '400', '100', '1200', '589', '-1', '1600', '1', '1070']
sec.sanctioned.amount (numeric, 2223 distinct): ['0', '50000', '100000', '30000', '40000', '200000', '15000', '25000', '10000', '300000']
sec.disbursed.amount (numeric, 2553 distinct): ['0', '50000', '100000', '200000', '40000', '300000', '30000', '500000', '150000', '400000']
primary.instal.amt (numeric, 28067 distinct): ['0', '1620', '1500', '1600', '2000', '2500', '1149', '1250', '1700', '1350']
sec.instal.amt (numeric, 1918 distinct): ['0', '2100', '1232', '1100', '1065', '5000', '1565', '1834', '2400', '50000']
new.accts.in.last.six.months (numeric, 26 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
delinquent.accts.in.last.six.months (numeric, 14 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '12']
average.acct.age (string, 192 distinct): ['0yrs 0mon', '0yrs 6mon', '0yrs 7mon', '0yrs 11mon', '0yrs 10mon', '1yrs 0mon', '0yrs 9mon', '0yrs 8mon', '1yrs 1mon', '0yrs 5mon']
credit.history.length (string, 294 distinct): ['0yrs 0mon', '0yrs 6mon', '2yrs 1mon', '0yrs 7mon', '2yrs 0mon', '1yrs 0mon', '1yrs 1mon', '0yrs 11mon', '0yrs 8mon', '0yrs 9mon']
no.of_inquiries (numeric, 25 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''

def parse_history_length(h: str) -> int:
    assert h.count('yrs ') == 1 and h.count('mon') == 1
    y, m = h.replace('mon', '').split('yrs ')
    return int(y) * 12 + int(m)


CONTEXT = "Vehicle Loan Default Applications for Indian Customers"
TARGET = CuratedTarget(raw_name="loan_default", new_name="Loan Default", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["uniqueid"]
FEATURES = [CuratedFeature(raw_name="date.of.birth", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="disbursaldate", new_name="Disbursal Date", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="credit.history.length", new_name="Credit History Length in Months",
                           feat_type=FeatureType.NUMERIC, processing_func=parse_history_length),
            CuratedFeature(raw_name='average.acct.age', new_name="Average Account Age in Months",
                           feat_type=FeatureType.NUMERIC, processing_func=parse_history_length),]
