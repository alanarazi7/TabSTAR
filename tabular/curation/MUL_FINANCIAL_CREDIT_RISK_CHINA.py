from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: credit_risk_china
====
Examples: 27522
====
URL: https://www.openml.org/search?type=data&id=46444
====
Description: Multi-classification assessment of bank (china) personal credit risk based on multi-source information fusion
====
Target Variable: five_categories (nominal, 5 distinct): ['Normal', 'Secondary', 'Loss', 'Concern', 'Suspicious']
====
Features:

customer_id (numeric, 27522 distinct): ['1', '18357', '18355', '18354', '18353', '18352', '18351', '18350', '18349', '18348']
type_of_loan_business (nominal, 22 distinct): ['Personal Housing Mortgage Loan (First hand)', 'Personal Commercial Housing Mortgage Loan (First hand)', 'Personal Housing Mortgage Loan (Second hand)', 'Personal Commercial Housing Mortgage Loan (Second hand)', 'Housing accumulation fund loan 3', 'Personal comprehensive consumer loans', 'Personal business loans', 'Personal insurance guarantees E credit', 'Personal auto loan', 'Personal rental loan for business purposes']
guarantee_the_balance (numeric, 20192 distinct): ['0.0', '0.01', '500000.0', '575983.57', '585583.36', '93670.97', '192500.0', '172675.02', '240000.0', '400000.0']
account_connection_amount (numeric, 24790 distinct): ['575983.57', '93670.97', '585583.36', '192500.0', '172675.02', '535654.39', '240000.0', '461002.1', '260000.0', '296358.49']
security_guarantee_amount (numeric, 20210 distinct): ['0.0', '0.8', '50000000.0', '9367097.0', '58558336.0', '57598357.0', '19250000.0', '17267502.0', '40000000.0', '147125000.0']
five-level_classification (nominal, 11 distinct): ['A1', 'A2', 'E', 'B1', 'D1', 'D2', 'B2', 'C1', 'C2', 'D3']
whether_interest_is_owed (nominal, 2 distinct): ['N', 'Y']
whether_self-service_loan (nominal, 2 distinct): ['N', 'Y']
type_of_guarantee (nominal, 18 distinct): ['General Enterprise Guarantee', 'Average house', 'Office buildings', 'Shops', 'Apartment', 'Guaranty by natural person', 'Guarantee Company Guarantee', 'Guarantee Insurance (Special Credit Granting)', 'Guarantee money for local currency third-party platform customers', 'Credit']
safety_coefficient (numeric, 9 distinct): ['100.0', '80.0', '70.0', '50.0', '60.0', '75.0', '77.0', '40.0']
collateral_value_(yuan) (numeric, 11578 distinct): ['188555192.45', '394663457.49', '88066944.5', '804141003.49', '262414761.01', '90605234.22', '114097337.99', '87817267.56', '213042029.21', '147557843.65']
guarantee_method (nominal, 5 distinct): ['Mortgage', 'Guarantee', 'Pledge', 'Credit']
date_code (nominal, 3 distinct): ['Y', 'M', 'D']
approval_deadline (numeric, 37 distinct): ['30', '20', '10', '25', '15', '28', '29', '24', '5', '27']
whether_devalue_account (nominal, 2 distinct): ['N', 'Y']
industry_category (nominal, 20 distinct): ['Public administration and social organization', 'Wholesale and Retail', 'Manufacturing', 'Education', ' Construction industry', 'Information transmission, computer services and software', 'Health, social security and social welfare industries', 'Residential services and other services', 'Mining industry', 'Transportation, warehousing and postal services']
down_payment_amount (numeric, 10037 distinct): ['0.0', '200000.0', '210000.0', '300000.0', '240000.0', '220000.0', '270000.0', '280000.0', '360000.0', '250000.0']
whether_personal_business_loan (nominal, 2 distinct): ['N', 'Y']
whether_interest_is_owed_(regulatory_standard) (nominal, 2 distinct): ['N', 'Y']
repayment_type (nominal, 3 distinct): ['Payment by installments', 'repayment of principal and interest at maturity']
installment_repayment_method_(numerical_type) (numeric, 3 distinct): ['1.0', '2.0']
installment_repayment_method_(discrete_type) (nominal, 3 distinct): ['Matching the principal and interest', 'Equal repayment of principal']
installment_repayment_cycle_(numerical_type) (nominal, 3 distinct): ['M01', 'M03']
repayment_cycle_(discrete_type) (nominal, 3 distinct): ['Month', 'Quarter']
number_of_houses (numeric, 5 distinct): ['1.0', '2.0', '4.0', '3.0']
month_property_costs (numeric, 3345 distinct): ['100.0', '200.0', '160.0', '152.34', '144.68', '150.0', '185.44', '135.35', '300.0', '94.0']
family_monthly_income (numeric, 2941 distinct): ['10000.0', '12000.0', '8000.0', '15000.0', '6000.0', '9000.0', '11000.0', '7000.0', '13000.0', '8500.0']
'''

CONTEXT = "Personal Chinese Credit Risk Assessment"
TARGET = CuratedTarget(raw_name="five_categories", new_name="Credit Risk", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []