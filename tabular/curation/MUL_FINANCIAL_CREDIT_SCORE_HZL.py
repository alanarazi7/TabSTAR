from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Credit_Score_Classification
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46441
====
Description: This dataset contains customer credit score information, which can be used for classification purposes.

**Target Variable**: Credit Score:
- **Poor** (0): Customers with a low credit score.
- **Standard** (1): Customers with an average credit score.
- **Good** (2): Customers with a high credit score.

**Features** include various attributes such as income, number of credit cards, loan information, and other financial indicators.
====
Target Variable: credit_score (nominal, 3 distinct): ['standard', 'poor', 'good']
====
Features:

id (string, 100000 distinct): ['0x1602', '0x19c88', '0x19caa', '0x19ca5', '0x19ca4', '0x19ca3', '0x19ca2', '0x19ca1', '0x19ca0', '0x19c9f']
customer_id (string, 12500 distinct): ['CUS_0xd40', 'CUS_0x9bf4', 'CUS_0x5ae3', 'CUS_0xbe9a', 'CUS_0x4874', 'CUS_0xc67b', 'CUS_0x8a64', 'CUS_0x35ea', 'CUS_0x5044', 'CUS_0x9dfd']
month (string, 8 distinct): ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
name (string, 10140 distinct): ['Langep', 'Stevex', 'Vaughanl', 'Jessicad', 'Raymondr', 'Danielz', 'Deepa Seetharamanm', 'Jessica Wohlt', 'Nate Raymondw', 'Nicko']
age (numeric, 1728 distinct): ['38.0', '28.0', '31.0', '26.0', '32.0', '36.0', '35.0', '25.0', '27.0', '39.0']
ssn (string, 12501 distinct): ['#F%$D@*&8', '078-73-5990', '486-78-3816', '750-67-7525', '903-50-0305', '376-28-6303', '194-93-5515', '442-30-8588', '362-78-8068', '221-76-9774']
occupation (string, 16 distinct): ['_______', 'Lawyer', 'Architect', 'Engineer', 'Scientist', 'Mechanic', 'Accountant', 'Developer', 'Media_Manager', 'Teacher']
annual_income (numeric, 13487 distinct): ['17816.75', '22434.16', '40341.16', '17273.83', '109945.32', '32543.38', '9141.63', '20867.67', '36585.12', '95596.35']
monthly_inhand_salary (numeric, 13236 distinct): ['6769.13', '6358.9567', '2295.0583', '6082.1875', '3080.555', '4387.2725', '5766.4917', '6639.56', '536.4312', '1315.5608']
num_bank_accounts (numeric, 942 distinct): ['6.0', '7.0', '8.0', '4.0', '5.0', '3.0', '9.0', '10.0', '1.0', '0.0']
num_credit_card (numeric, 1179 distinct): ['5.0', '7.0', '6.0', '4.0', '3.0', '8.0', '10.0', '9.0', '2.0', '1.0']
interest_rate (numeric, 1750 distinct): ['8.0', '5.0', '6.0', '12.0', '10.0', '9.0', '7.0', '11.0', '18.0', '15.0']
num_of_loan (numeric, 413 distinct): ['3.0', '2.0', '4.0', '0.0', '1.0', '6.0', '7.0', '5.0', '100.0', '9.0']
type_of_loan (string, 6261 distinct): ['Not Specified', 'Credit-Builder Loan', 'Personal Loan', 'Debt Consolidation Loan', 'Student Loan', 'Payday Loan', 'Mortgage Loan', 'Auto Loan', 'Home Equity Loan', 'Personal Loan, and Student Loan']
delay_from_due_date (numeric, 68 distinct): ['15', '13', '8', '14', '10', '7', '9', '11', '12', '6']
num_of_delayed_payment (numeric, 709 distinct): ['19.0', '17.0', '16.0', '10.0', '15.0', '18.0', '20.0', '12.0', '9.0', '8.0']
changed_credit_limit (numeric, 4376 distinct): ['8.22', '11.5', '11.32', '10.06', '7.35', '8.23', '11.49', '7.33', '7.69', '9.25']
num_credit_inquiries (numeric, 1224 distinct): ['4.0', '3.0', '6.0', '7.0', '2.0', '8.0', '1.0', '0.0', '5.0', '9.0']
credit_mix (string, 4 distinct): ['Standard', 'Good', '_', 'Bad']
outstanding_debt (numeric, 12203 distinct): ['1109.03', '1151.7', '1360.45', '460.46', '1058.13', '1466.97', '328.81', '198.06', '762.58', '330.98']
credit_utilization_ratio (numeric, 100000 distinct): ['26.8226', '28.3279', '30.0166', '25.4788', '33.9338', '30.381', '34.8788', '36.5261', '33.1094', '37.0896']
credit_history_age (string, 405 distinct): ['15 Years and 11 Months', '19 Years and 4 Months', '19 Years and 5 Months', '17 Years and 11 Months', '19 Years and 3 Months', '17 Years and 9 Months', '15 Years and 10 Months', '17 Years and 10 Months', '15 Years and 9 Months', '18 Years and 3 Months']
payment_of_min_amount (string, 3 distinct): ['Yes', 'No', 'NM']
total_emi_per_month (numeric, 14950 distinct): ['0.0', '49.5749', '73.5334', '22.9608', '38.6611', '56.9768', '188.2717', '41.9024', '60.8172', '66.7829']
amount_invested_monthly (numeric, 91050 distinct): ['10000.0', '0.0', '80.4153', '36.6624', '89.7385', '59.9373', '165.1807', '62.0308', '215.5771', '44.6114']
payment_behaviour (string, 7 distinct): ['Low_spent_Small_value_payments', 'High_spent_Medium_value_payments', 'Low_spent_Medium_value_payments', 'High_spent_Large_value_payments', 'High_spent_Small_value_payments', 'Low_spent_Large_value_payments', '!@9#%8']
monthly_balance (numeric, 98793 distinct): ['3.333333333333333e+26', '312.4941', '415.3253', '252.0849', '254.9709', '250.0932', '289.7551', '260.6258', '606.8304', '111.9905']
'''

CONTEXT = "Customer Credit Score Classification"
TARGET = CuratedTarget(raw_name="credit_score", new_name="Credit Score", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["id"]
FEATURES = [CuratedFeature(raw_name="credit_mix", new_name="Credit Mix", value_mapping={'_': 'Unknown'}),
            CuratedFeature(raw_name="occupation", value_mapping={'_______': 'Unknown'})]
