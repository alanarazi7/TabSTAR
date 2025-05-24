from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Personal-Loan-Modeling
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=43826
====
Description: Context
This case is about a bank (Thera Bank) whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9 success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.
Content
The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6) accepted the personal loan that was offered to them in the earlier campaign.
There are no empty or (NaN) values in the dataset. The dataset has a mix of numerical and categorical attributes, but all categorical data are represented with numbers. Moreover, Some of the predictor variables are heavily skewed (long - tailed), making the data pre-processing an interesting yet not too challenging aspect of the data.
====
Target Variable: Personal_Loan (numeric, 2 distinct): ['0', '1']
====
Features:

Age (numeric, 45 distinct): ['35', '43', '52', '54', '58', '50', '41', '30', '56', '34']
Experience (numeric, 47 distinct): ['32', '20', '9', '5', '23', '35', '25', '28', '18', '19']
Income (numeric, 162 distinct): ['44', '38', '81', '41', '39', '40', '42', '83', '43', '45']
ZIP_Code (numeric, 467 distinct): ['94720', '94305', '95616', '90095', '93106', '93943', '92037', '91320', '91711', '94025']
Family (numeric, 4 distinct): ['1', '2', '4', '3']
CCAvg (numeric, 108 distinct): ['0.3', '1.0', '0.2', '2.0', '0.8', '0.1', '0.4', '1.5', '0.7', '0.5']
Education (numeric, 3 distinct): ['1', '3', '2']
Mortgage (numeric, 347 distinct): ['0', '98', '119', '89', '91', '103', '83', '102', '90', '78']
Securities_Account (numeric, 2 distinct): ['0', '1']
CD_Account (numeric, 2 distinct): ['0', '1']
Online (numeric, 2 distinct): ['1', '0']
CreditCard (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Liability Bank Customers"
TARGET = CuratedTarget(raw_name='Personal_Loan', new_name='Customer Response for Personal Loan Campaign',
                       task_type=SupervisedTask.BINARY, label_mapping={'0': 'No', '1': 'Yes'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name='CCAvg', new_name='Credit Card Average Spending'),
            CuratedFeature(raw_name='Securities_Account'),
            CuratedFeature(raw_name='CD_Account', new_name='Has Certificate of Deposit Account'),
            CuratedFeature(raw_name='Online', new_name='Is Online Customer'),
            CuratedFeature(raw_name='CreditCard', new_name='Has Credit Card')]
