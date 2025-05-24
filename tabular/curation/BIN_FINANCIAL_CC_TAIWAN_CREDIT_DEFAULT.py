from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Default-of-Credit-Card-Clients-Dataset
====
Examples: 30000
====
URL: https://www.openml.org/search?type=data&id=43435
====
Description: Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. 
Content
There are 25 variables:

ID: ID of each client
LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
SEX: Gender (1=male, 2=female)
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status (1=married, 2=single, 3=others)
AGE: Age in years
PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months,  8=payment delay for eight months, 9=payment delay for nine months and above)
PAY_2: Repayment status in August, 2005 (scale same as above)
PAY_3: Repayment status in July, 2005 (scale same as above)
PAY_4: Repayment status in June, 2005 (scale same as above)
PAY_5: Repayment status in May, 2005 (scale same as above)
PAY_6: Repayment status in April, 2005 (scale same as above)
BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
default.payment.next.month: Default payment (1=yes, 0=no)

Inspiration
Some ideas for exploration:

How does the probability of default payment vary by categories of different demographic variables?
Which variables are the strongest predictors of default payment?

Acknowledgements
Any publications based on this dataset should acknowledge the following: 
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
The original dataset can be found here at the UCI Machine Learning Repository.
====
Target Variable: default.payment.next.month (numeric, 2 distinct): ['0', '1']
====
Features:

LIMIT_BAL (numeric, 81 distinct): ['50000.0', '20000.0', '30000.0', '80000.0', '200000.0', '150000.0', '100000.0', '180000.0', '360000.0', '60000.0']
SEX (numeric, 2 distinct): ['2', '1']
EDUCATION (numeric, 7 distinct): ['2', '1', '3', '5', '4', '6', '0']
MARRIAGE (numeric, 4 distinct): ['2', '1', '3', '0']
AGE (numeric, 56 distinct): ['29', '27', '28', '30', '26', '31', '25', '34', '32', '33']
PAY_0 (numeric, 11 distinct): ['0', '-1', '1', '-2', '2', '3', '4', '5', '8', '6']
PAY_2 (numeric, 11 distinct): ['0', '-1', '2', '-2', '3', '4', '1', '5', '7', '6']
PAY_3 (numeric, 11 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '6', '5', '1']
PAY_4 (numeric, 11 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '5', '6', '1']
PAY_5 (numeric, 10 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '5', '6', '8']
PAY_6 (numeric, 10 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '6', '5', '8']
BILL_AMT1 (numeric, 22723 distinct): ['0.0', '390.0', '780.0', '326.0', '316.0', '2500.0', '396.0', '2400.0', '416.0', '500.0']
BILL_AMT2 (numeric, 22346 distinct): ['0.0', '390.0', '326.0', '780.0', '316.0', '396.0', '2500.0', '2400.0', '-200.0', '416.0']
BILL_AMT3 (numeric, 22026 distinct): ['0.0', '390.0', '780.0', '326.0', '316.0', '396.0', '2500.0', '2400.0', '416.0', '200.0']
BILL_AMT4 (numeric, 21548 distinct): ['0.0', '390.0', '780.0', '316.0', '326.0', '396.0', '2400.0', '150.0', '2500.0', '416.0']
BILL_AMT5 (numeric, 21010 distinct): ['0.0', '390.0', '780.0', '316.0', '326.0', '150.0', '396.0', '2400.0', '2500.0', '416.0']
BILL_AMT6 (numeric, 20604 distinct): ['0.0', '390.0', '780.0', '150.0', '316.0', '326.0', '396.0', '416.0', '-18.0', '2400.0']
PAY_AMT1 (numeric, 7943 distinct): ['0.0', '2000.0', '3000.0', '5000.0', '1500.0', '4000.0', '10000.0', '1000.0', '2500.0', '6000.0']
PAY_AMT2 (numeric, 7899 distinct): ['0.0', '2000.0', '3000.0', '5000.0', '1000.0', '1500.0', '4000.0', '10000.0', '6000.0', '2500.0']
PAY_AMT3 (numeric, 7518 distinct): ['0.0', '2000.0', '1000.0', '3000.0', '5000.0', '1500.0', '4000.0', '10000.0', '1200.0', '6000.0']
PAY_AMT4 (numeric, 6937 distinct): ['0.0', '1000.0', '2000.0', '3000.0', '5000.0', '1500.0', '4000.0', '10000.0', '2500.0', '500.0']
PAY_AMT5 (numeric, 6897 distinct): ['0.0', '1000.0', '2000.0', '3000.0', '5000.0', '1500.0', '4000.0', '10000.0', '500.0', '6000.0']
PAY_AMT6 (numeric, 6939 distinct): ['0.0', '1000.0', '2000.0', '3000.0', '5000.0', '1500.0', '4000.0', '10000.0', '500.0', '6000.0']
'''

CONTEXT = "Taiwan Credit Card Clients during 2005"
TARGET = CuratedTarget(raw_name='default.payment.next.month', new_name='Taiwan Credit Card Default',
                       task_type=SupervisedTask.BINARY, label_mapping={'0': 'No', '1': 'Yes'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name='LIMIT_BAL', new_name='Amount of Credit Limit in NT Dollars',),
            CuratedFeature(raw_name='SEX', new_name='Sex', value_mapping={'1': 'Male', '2': 'Female'}),
            CuratedFeature(raw_name='EDUCATION', new_name='Education Level',
                           value_mapping={'1': 'Graduate School', '2': 'University', '3': 'High School', '4': 'Others',
                                          '5': 'Unknown 5', '6': 'Unknown 6', '0': 'Unknown 0'}),
            CuratedFeature(raw_name='MARRIAGE', new_name='Marital Status',
                            value_mapping={'1': 'Married', '2': 'Single', '3': 'Others', '0': 'Unknown'}),
            CuratedFeature(raw_name='AGE', new_name='Age in Years',),
            CuratedFeature(raw_name='PAY_0', new_name='Repayment Status last month'),
            CuratedFeature(raw_name='PAY_2', new_name='Repayment Status 2 months before'),
            CuratedFeature(raw_name='PAY_3', new_name='Repayment Status 3 months before'),
            CuratedFeature(raw_name='PAY_4', new_name='Repayment Status 4 months before'),
            CuratedFeature(raw_name='PAY_5', new_name='Repayment Status 5 months before'),
            CuratedFeature(raw_name='PAY_6', new_name='Repayment Status 6 months before'),
            CuratedFeature(raw_name='BILL_AMT1', new_name='Amount of Bill Statement last month',),
            CuratedFeature(raw_name='BILL_AMT2', new_name='Amount of Bill Statement 2 months before',),
            CuratedFeature(raw_name='BILL_AMT3', new_name='Amount of Bill Statement 3 months before',),
            CuratedFeature(raw_name='BILL_AMT4', new_name='Amount of Bill Statement 4 months before',),
            CuratedFeature(raw_name='BILL_AMT5', new_name='Amount of Bill Statement 5 months before',),
            CuratedFeature(raw_name='BILL_AMT6', new_name='Amount of Bill Statement 6 months before',),
            CuratedFeature(raw_name='PAY_AMT1', new_name='Amount of Previous Payment last month',),
            CuratedFeature(raw_name='PAY_AMT2', new_name='Amount of Previous Payment 2 months before',),
            CuratedFeature(raw_name='PAY_AMT3', new_name='Amount of Previous Payment 3 months before',),
            CuratedFeature(raw_name='PAY_AMT4', new_name='Amount of Previous Payment 4 months before',),
            CuratedFeature(raw_name='PAY_AMT5', new_name='Amount of Previous Payment 5 months before',),
            CuratedFeature(raw_name='PAY_AMT6', new_name='Amount of Previous Payment 6 months before',)
            ]