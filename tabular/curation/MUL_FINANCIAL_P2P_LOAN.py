from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: Online-P2P-Lending
====
URL: https://www.openml.org/search?type=data&id=43502
====
Description: P2P Lending
I concatenated historical loans from both Prosper and Lending Club 2013 - 2018.  Currently only the summary of the loan (terms, origination date, loan amount, status, etc) are up but detailed lender data will come soon.  The columns are matched up as accurately as possible but there are estimated columns, see below for more info.
Content
To come.
====
Features:

loan_number (numeric, 2874088 distinct): ['847974', '1056947', '367050', '709970', '644351', '382607', '555039', '617222', '783966', '983823']
amount_borrowed (numeric, 6526 distinct): ['10000.0', '15000.0', '20000.0', '12000.0', '5000.0', '35000.0', '25000.0', '8000.0', '6000.0', '30000.0']
term (numeric, 3 distinct): ['36', '60', '12']
borrower_rate (numeric, 863 distinct): ['0.1199', '0.0532', '0.1099', '0.1399', '0.1149', '0.1699', '0.1299', '0.0789', '0.0917', '0.1505']
installment (numeric, 157310 distinct): ['325.9091', '301.15', '332.1', '327.34', '361.38', '602.3', '451.73', '329.72', '166.05', '498.15']
grade (string, 7 distinct): ['C', 'B', 'D', 'A', 'E', 'F', 'G']
origination_date (string, 1459 distinct): ['2016-03-01T00:00', '2015-10-01T00:00', '2018-05-01T00:00', '2015-07-01T00:00', '2015-12-01T00:00', '2017-08-01T00:00', '2018-04-01T00:00', '2017-11-01T00:00', '2018-06-01T00:00', '2017-09-01T00:00']
listing_title (string, 15 distinct): ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'small_business', 'car', 'vacation', 'moving']
principal_balance (numeric, 568401 distinct): ['0.0', '10000.0', '20000.0', '15000.0', '12000.0', '5000.0', '40000.0', '25000.0', '35000.0', '30000.0']
principal_paid (numeric, 641810 distinct): ['10000.0', '15000.0', '20000.0', '12000.0', '5000.0', '8000.0', '6000.0', '35000.0', '25000.0', '0.0']
interest_paid (numeric, 686488 distinct): ['0.0', '1431.12', '1784.23', '1955.4', '956.78', '1977.77', '2128.02', '2862.28', '1264.46', '1517.36']
late_fees_paid (numeric, 22590 distinct): ['0.0', '15.0', '30.0', '45.0', '60.0', '75.0', '90.0', '105.0', '120.0', '135.0']
debt_sale_proceeds_received (numeric, 100087 distinct): ['0.0', '50.0', '100.0', '200.0', '150.0', '25.0', '300.0', '75.0', '400.0', '250.0']
last_payment_date (string, 2564 distinct): ['2018-06-01T00:00', '2018-07-01T00:00', '2018-03-01T00:00', '2018-05-01T00:00', '2018-01-01T00:00', '2017-03-01T00:00', '2018-04-01T00:00', '2017-10-01T00:00', '2017-08-01T00:00', '2017-11-01T00:00']
next_payment_due_date (string, 2210 distinct): ['2018-07-01T00:00', '2018-08-01T00:00', '2018-04-01T00:00', '2018-06-01T00:00', '2018-02-01T00:00', '2017-04-01T00:00', '2018-05-01T00:00', '2017-11-01T00:00', '2017-09-01T00:00', '2017-12-01T00:00']
days_past_due (numeric, 1635 distinct): ['0', '60', '29', '90', '122', '151', '363', '516', '121', '302']
loan_status_description (string, 5 distinct): ['CURRENT', 'COMPLETED', 'CHARGEOFF', 'DEFAULTED', 'CANCELLED']
data_source (string, 2 distinct): ['Lending Club', 'Prosper']
'''

CONTEXT = "P2P Lending"
TARGET = CuratedTarget(raw_name="loan_status_description", new_name="P2P Loan Status Description",
                       task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["loan_number"]
FEATURES = [CuratedFeature(raw_name="origination_date", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="last_payment_date", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="next_payment_due_date", feat_type=FeatureType.DATE)]
