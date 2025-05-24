from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Give-Me-Some-Credit
====
Examples: 150000
====
URL: https://www.openml.org/search?type=data&id=45577
====
Description: Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.

yeah

## Description

Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 

Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

The goal of this competition is to build a model that borrowers can use to help make the best financial decisions.

Historical data are provided on 250,000 borrowers and the prize pool is $5,000 ($3,000 for first, $1,500 for second and $500 for third).

## Features

| Variable Name                        | Description                                                                                                                                              | Type       |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| SeriousDlqin2yrs                     | Person experienced 90 days past due delinquency or worse                                                                                                 | Y/N        |
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits | percentage |
| age                                  | Age of borrower in years                                                                                                                                 | integer    |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years.                                                                  | integer    |
| DebtRatio                            | Monthly debt payments, alimony,living costs divided by monthy gross income                                                                               | percentage |
| MonthlyIncome                        | Monthly income                                                                                                                                           | real       |
| NumberOfOpenCreditLinesAndLoans      | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)                                                     | integer    |
| NumberOfTimes90DaysLate              | Number of times borrower has been 90 days or more past due.                                                                                              | integer    |
| NumberRealEstateLoansOrLines         | Number of mortgage and real estate loans including home equity lines of credit                                                                           | integer    |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years.                                                                  | integer    |
| NumberOfDependents                   | Number of dependents in family excluding themselves (spouse, children etc.)                                                                              | integer    |

**Note:** This is the training part of the Kaggle competition going by the dataset name hosted [here](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview).
====
Target Variable: SeriousDlqin2yrs (nominal, 2 distinct): ['0', '1']
====
Features:

RevolvingUtilizationOfUnsecuredLines (numeric, 125728 distinct): ['0.0', '1.0', '1.0', '0.9501', '0.008', '0.9541', '0.7131', '0.7964', '0.988', '0.994']
age (numeric, 86 distinct): ['49', '48', '50', '47', '63', '46', '53', '51', '52', '56']
NumberOfTime30-59DaysPastDueNotWorse (numeric, 16 distinct): ['0', '1', '2', '3', '4', '5', '98', '6', '7', '8']
DebtRatio (numeric, 114194 distinct): ['0.0', '1.0', '4.0', '2.0', '3.0', '5.0', '9.0', '10.0', '7.0', '13.0']
MonthlyIncome (numeric, 13595 distinct): ['5000.0', '4000.0', '6000.0', '3000.0', '0.0', '2500.0', '10000.0', '3500.0', '4500.0', '7000.0']
NumberOfOpenCreditLinesAndLoans (numeric, 58 distinct): ['6', '7', '5', '8', '4', '9', '10', '3', '11', '12']
NumberOfTimes90DaysLate (numeric, 19 distinct): ['0', '1', '2', '3', '4', '98', '5', '6', '7', '8']
NumberRealEstateLoansOrLines (numeric, 28 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NumberOfTime60-89DaysPastDueNotWorse (numeric, 13 distinct): ['0', '1', '2', '3', '98', '4', '5', '6', '7', '96']
NumberOfDependents (numeric, 14 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '10.0']
'''

CONTEXT = "Credit Scoring for Financial Distress Prediction"
TARGET = CuratedTarget(raw_name="SeriousDlqin2yrs", new_name="Serious Delinquency in 2 years",
                       task_type=SupervisedTask.BINARY, label_mapping={'0': 'No', '1': 'Yes'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="RevolvingUtilizationOfUnsecuredLines", new_name="Revolving Utilization of Unsecured Lines"),
            CuratedFeature(raw_name="age", new_name="Age of borrower"),
            CuratedFeature(raw_name="NumberOfTime30-59DaysPastDueNotWorse", new_name="Number of Times 30-59 Days Past Due Not Worse"),
            CuratedFeature(raw_name="DebtRatio", new_name="Debt Ratio"),
            CuratedFeature(raw_name="MonthlyIncome", new_name="Monthly Income"),
            CuratedFeature(raw_name="NumberOfOpenCreditLinesAndLoans", new_name="Number of Open Credit Lines and Loans"),
            CuratedFeature(raw_name="NumberOfTimes90DaysLate", new_name="Number of Times 90 Days Late"),
            CuratedFeature(raw_name="NumberRealEstateLoansOrLines", new_name="Number of Real Estate Loans or Lines including home equity"),
            CuratedFeature(raw_name="NumberOfTime60-89DaysPastDueNotWorse", new_name="Number of Times 60-89 Days Past Due Not Worse"),
            CuratedFeature(raw_name="NumberOfDependents", new_name="Number of Dependents"),
]
