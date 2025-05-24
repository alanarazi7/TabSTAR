from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Credit-Card-Dataset-for-Clustering
====
URL: https://www.openml.org/search?type=data&id=43618
====
Description: This case requires to develop a customer segmentation to define marketing strategy. The
sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.
Following is the Data Dictionary for Credit Card dataset :-
CUSTID : Identification of Credit Card holder (Categorical)
BALANCE : Balance amount left in their account to make purchases (
BALANCEFREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
PURCHASES : Amount of purchases made from account
ONEOFFPURCHASES : Maximum purchase amount done in one-go
INSTALLMENTSPURCHASES : Amount of purchase done in installment
CASHADVANCE : Cash in advance given by the user
PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
PURCHASESTRX : Numbe of purchase transactions made
CREDITLIMIT : Limit of Credit Card for user
PAYMENTS : Amount of Payment done by user
MINIMUM_PAYMENTS : Minimum amount of payments made by user
PRCFULLPAYMENT : Percent of full payment paid by user
TENURE : Tenure of credit card service for user
====
Features:

BALANCE (numeric, 8871 distinct): ['0.0', '40.9007', '1213.5513', '1253.1883', '5058.2996', '296.9059', '1084.6526', '237.1984', '1636.5183', '468.8514']
BALANCE_FREQUENCY (numeric, 43 distinct): ['1.0', '0.9091', '0.8182', '0.7273', '0.5455', '0.6364', '0.4545', '0.3636', '0.2727', '0.1818']
PURCHASES (numeric, 6203 distinct): ['0.0', '45.65', '60.0', '150.0', '300.0', '200.0', '100.0', '450.0', '50.0', '600.0']
ONEOFF_PURCHASES (numeric, 4014 distinct): ['0.0', '45.65', '50.0', '200.0', '60.0', '100.0', '150.0', '70.0', '1000.0', '250.0']
INSTALLMENTS_PURCHASES (numeric, 4452 distinct): ['0.0', '300.0', '200.0', '100.0', '150.0', '125.0', '75.0', '350.0', '450.0', '500.0']
CASH_ADVANCE (numeric, 4323 distinct): ['0.0', '495.4258', '1486.2433', '855.2328', '3767.1047', '291.6085', '38.6906', '521.6644', '1974.203', '2462.1008']
PURCHASES_FREQUENCY (numeric, 47 distinct): ['1.0', '0.0', '0.0833', '0.9167', '0.5', '0.1667', '0.8333', '0.3333', '0.25', '0.5833']
ONEOFF_PURCHASES_FREQUENCY (numeric, 47 distinct): ['0.0', '0.0833', '0.1667', '1.0', '0.25', '0.3333', '0.4167', '0.5', '0.5833', '0.6667']
PURCHASES_INSTALLMENTS_FREQUENCY (numeric, 47 distinct): ['0.0', '1.0', '0.4167', '0.9167', '0.8333', '0.5', '0.1667', '0.6667', '0.75', '0.0833']
CASH_ADVANCE_FREQUENCY (numeric, 54 distinct): ['0.0', '0.0833', '0.1667', '0.25', '0.3333', '0.4167', '0.5', '0.5833', '0.6667', '0.0909']
CASH_ADVANCE_TRX (numeric, 65 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
PURCHASES_TRX (numeric, 173 distinct): ['0', '1', '12', '2', '6', '3', '4', '7', '5', '8']
CREDIT_LIMIT (numeric, 206 distinct): ['3000.0', '1500.0', '1200.0', '1000.0', '2500.0', '4000.0', '6000.0', '5000.0', '2000.0', '7500.0']
PAYMENTS (numeric, 8711 distinct): ['0.0', '201.8021', '398.3164', '826.0367', '2571.5732', '1903.2796', '454.8885', '956.0287', '4560.7757', '1825.35']
MINIMUM_PAYMENTS (numeric, 8949 distinct): ['299.3519', '342.2865', '184.4647', '276.4861', '309.1409', '354.2811', '216.0904', '277.5467', '150.3171', '1600.2692']
PRC_FULL_PAYMENT (numeric, 47 distinct): ['0.0', '1.0', '0.0833', '0.1667', '0.5', '0.25', '0.0909', '0.3333', '0.1', '0.2']
TENURE (numeric, 7 distinct): ['12', '11', '10', '6', '8', '7', '9']
'''

CONTEXT = "Credit Card User Segmentation"
TARGET = CuratedTarget(raw_name="CREDIT_LIMIT", new_name="Credit Card User Credit Limit",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
