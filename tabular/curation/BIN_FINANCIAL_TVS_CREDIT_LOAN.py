from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: TVS_Loan_Default
====
Examples: 119528
====
URL: https://www.openml.org/search?type=data&id=43743
====
Description: Personal Loan product is an unsecured loan therefore it is vital to assess the risk of the customers by checking their credit worthiness. This must be done to prevent loan defaults.
The objective is to build a Risk model using the dataset which will assess the risk of a customer defaulting after cross-selling the Personal Loan.
Column Descriptions:
V1:  Customer ID
V2:  If a customer has bounced in first EMI (1 : Bounced, 0 : Not bounced)
V3:  Number of times bounced in recent 12 months
V4:  Maximum MOB (Month of business with TVS Credit)
V5:  Number of times bounced while repaying the loan
V6:  EMI
V7:  Loan Amount
V8:  Tenure
V9:  Dealer codes from where customer has purchased the Two wheeler
V10:  Product code of Two wheeler (MC : Motorcycle , MO : Moped, SC : Scooter)
V11:  No of advance EMI paid
V12:  Rate of interest
V13:  Gender (Male/Female)
V14:  Employment type (HOUSEWIFE : housewife, SELF : Self-employed, SAL : Salaried, PENS : Pensioner, STUDENT : Student)
V15:  Resident type of customer
V16:  Date of birth
V17:  Age at which customer has taken the loan
V18:  Number of loans
V19:  Number of secured loans
V20:  Number of unsecured loans
V21:  Maximum amount sanctioned in the Live loans
V22:  Number of new loans in last 3 months
V23:  Total sanctioned amount in the secured Loans which are Live
V24:  Total sanctioned amount in the unsecured Loans which are Live
V25:  Maximum amount sanctioned for any Two wheeler loan
V26:  Time since last Personal loan taken (in months)
V27:  Time since first consumer durables loan taken (in months)
V28:  Number of times 30 days past due in last 6 months
V29:  Number of times 60 days past due in last 6 months
V30:  Number of times 90 days past due in last 3 months
V31:  Tier ; (Customers geographical location)
V32:  Target variable ( 1: Defaulters / 0: Non-Defaulters)
====
Features:

V1 (numeric, 119528 distinct): ['1', '79681', '79693', '79692', '79691', '79690', '79689', '79688', '79687', '79686']
V2 (numeric, 2 distinct): ['0', '1']
V3 (numeric, 13 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '11']
V4 (numeric, 35 distinct): ['20.0', '15.0', '21.0', '14.0', '10.0', '16.0', '19.0', '13.0', '22.0', '12.0']
V5 (numeric, 25 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V6 (numeric, 3293 distinct): ['2912.0', '2475.0', '2500.0', '3000.0', '2625.0', '2000.0', '2289.0', '3151.0', '1481.0', '2520.0']
V7 (numeric, 6290 distinct): ['30000.0', '45900.0', '42900.0', '48900.0', '25000.0', '50900.0', '47900.0', '46900.0', '28900.0', '32000.0']
V8 (numeric, 32 distinct): ['24.0', '18.0', '12.0', '36.0', '15.0', '30.0', '21.0', '10.0', '20.0', '25.0']
V9 (numeric, 3251 distinct): ['1740.0', '1004.0', '1241.0', '1396.0', '1384.0', '1079.0', '1151.0', '1165.0', '1615.0', '1012.0']
V10 (string, 6 distinct): ['SC', 'MO', 'MC', 'TL', 'RETOP']
V11 (numeric, 8 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0']
V12 (numeric, 1025 distinct): ['3.99', '13.0', '14.09', '13.5', '12.5', '8.88', '6.99', '12.0', '7.99', '14.0']
V13 (string, 3 distinct): ['MALE', 'FEMALE']
V14 (string, 6 distinct): ['SELF', 'SAL', 'HOUSEWIFE', 'STUDENT', 'PENS']
V15 (string, 4 distinct): ['OWNED', 'RENT', 'OWENED BY OFFICE']
V16 (string, 13487 distinct): ['01-01-1976', '01-01-1978', '01-01-1977', '01-01-1975', '01-01-1981', '01-01-1973', '01-01-1983', '01-01-1986', '01-01-1980', '01-01-1985']
V17 (numeric, 50 distinct): ['31.0', '30.0', '40.0', '32.0', '33.0', '26.0', '29.0', '35.0', '27.0', '34.0']
V18 (numeric, 121 distinct): ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
V19 (numeric, 109 distinct): ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
V20 (numeric, 47 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
V21 (numeric, 9901 distinct): ['100000.0', '50000.0', '200000.0', '300000.0', '500000.0', '150000.0', '40000.0', '25000.0', '400000.0', '60000.0']
V22 (numeric, 1 distinct): ['0']
V23 (numeric, 7273 distinct): ['100000.0', '50000.0', '500000.0', '300000.0', '30000.0', '25000.0', '40000.0', '400000.0', '200000.0', '1000000.0']
V24 (numeric, 10415 distinct): ['100000.0', '50000.0', '200000.0', '150000.0', '300000.0', '40000.0', '400000.0', '25000.0', '500000.0', '60000.0']
V25 (numeric, 8682 distinct): ['30000.0', '45900.0', '48900.0', '50900.0', '47900.0', '46900.0', '28900.0', '44900.0', '42900.0', '45000.0']
V26 (numeric, 225 distinct): ['31.0', '25.0', '20.0', '24.0', '22.0', '27.0', '21.0', '26.0', '28.0', '23.0']
V27 (numeric, 247 distinct): ['30.0', '25.0', '31.0', '19.0', '24.0', '28.0', '22.0', '26.0', '23.0', '18.0']
V28 (numeric, 86 distinct): ['0', '1', '2', '6', '3', '5', '4', '7', '8', '12']
V29 (numeric, 81 distinct): ['0', '1', '6', '2', '3', '5', '4', '7', '12', '8']
V30 (numeric, 46 distinct): ['0', '3', '1', '2', '4', '6', '5', '9', '7', '8']
V31 (string, 4 distinct): ['TIER 4', 'TIER 3', 'TIER 1', 'TIER 2']
V32 (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Personal Load Default LTS"
TARGET = CuratedTarget(raw_name="V32", new_name="Defaulters", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'Non-Defaulters', '1': 'Defaulters'})
COLS_TO_DROP = ["V1", "V22"]
FEATURES = [
    CuratedFeature(raw_name="V2", new_name="Bounced in First EMI"),
    CuratedFeature(raw_name="V3", new_name="Number of times bounced in recent 12 months"),
    CuratedFeature(raw_name="V4", new_name="Maximum MOB (Month of business with TVS Credit)"),
    CuratedFeature(raw_name="V5", new_name="Number of times bounced while repaying the loan"),
    CuratedFeature(raw_name="V6", new_name="EMI"),
    CuratedFeature(raw_name="V7", new_name="Loan Amount"),
    CuratedFeature(raw_name="V8", new_name="Tenure"),
    CuratedFeature(raw_name="V9", new_name="Dealer codes from where customer has purchased the Two wheeler"),
    CuratedFeature(raw_name="V10", new_name="Product code of Two wheeler",
                   value_mapping={'SC': 'Scooter', 'MO': 'Moped', 'MC': 'Motorcycle'}),
    CuratedFeature(raw_name="V11", new_name="No of advance EMI paid"),
    CuratedFeature(raw_name="V12", new_name="Rate of interest"),
    CuratedFeature(raw_name="V13", new_name="Gender"),
    CuratedFeature(raw_name="V14", new_name="Employment type",
                   value_mapping={'SELF': 'Self-employed', 'SAL': 'Salaried', 'HOUSEWIFE': 'Housewife',
                                  'STUDENT': 'Student', 'PENS': 'Pensioner'}),
    CuratedFeature(raw_name="V15", new_name="Resident type of customer"),
    CuratedFeature(raw_name="V16", new_name="Date of birth", feat_type=FeatureType.DATE),
    CuratedFeature(raw_name="V17", new_name="Age at which customer has taken the loan"),
    CuratedFeature(raw_name="V18", new_name="Number of loans"),
    CuratedFeature(raw_name="V19", new_name="Number of secured loans"),
    CuratedFeature(raw_name="V20", new_name="Number of unsecured loans"),
    CuratedFeature(raw_name="V21", new_name="Maximum amount sanctioned in the Live loans"),
    CuratedFeature(raw_name="V23", new_name="Total sanctioned amount in the secured Loans which are Live"),
    CuratedFeature(raw_name="V24", new_name="Total sanctioned amount in the unsecured Loans which are Live"),
    CuratedFeature(raw_name="V25", new_name="Maximum amount sanctioned for any Two wheeler loan"),
    CuratedFeature(raw_name="V26", new_name="Time since last Personal loan taken (in months)"),
    CuratedFeature(raw_name="V27", new_name="Time since first consumer durables loan taken (in months)"),
    CuratedFeature(raw_name="V28", new_name="Number of times 30 days past due in last 6 months"),
    CuratedFeature(raw_name="V29", new_name="Number of times 60 days past due in last 6 months"),
    CuratedFeature(raw_name="V30", new_name="Number of times 90 days past due in last 3 months"),
    CuratedFeature(raw_name="V31", new_name="Tier"),
]