from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Municipal-Debt-Risk-Analysis
====
Examples: 138509
====
URL: https://www.openml.org/search?type=data&id=43838
====
Description: Context
This data has been extracted from the billing systems of 8 Municipalities in South Africa over a 2 year period and summarised according to their total amount billed versus the total amount paid. For each account there is an indicator of whether that account resulted in a Bad Debt.
This is a Classification exercise with the aim of finding out whether it is feasible to determine the probability of an account becoming a Bad Debt so that it will be possible to forecast the number (and value) of accounts that are at risk of developing into a Bad Debt.
Content
AccCategoryID: (Account Category ID) The numeric link in the database to the Account Category
AccCategory: (Account Category) A classification of the type of account
AccCategoryAbbr: (Account Category Abbreviation) An abbreviation of the classification of the type of account - to be used for One-hot encoding
PropertyValue: (Property Value) The market value of the property
PropertySize: (Property Size) The size of the property in square metres
TotalBilling: (Total Billing) The total amount billed to the account for all services
AverageBilling: (Average Billing) The average amount billed to the account for all services
TotalReceipting: (Total Receipting) The total amount receipted to the account for all services
AverageReceipting: (Average Receipting) The average amount receipted to the account for all services
TotalDebt: (Total Debt) The Total Debt that is at 90 days or more
TotalWriteOff: (Total Write Off) The Total amount of debt that has been written off
CollectionRatio: (Collection Ratio) The ratio between the Total Receipting and Total Billing (ie. Total Receipting/Total Billing)
DebtBillingRatio: (Billing Debt Ratio) The ratio between the Total Debt and Total Billing (ie. (Total Debt + Total Write Off)/Total Billing)
TotalElectricityBill: (Total Electricity Bill) The total amount billed for electricity. This field was put in place because it is used as a means to recover debt - ie. If an amount is outstanding for any service the municipality has the right to cut a consumer's electricity connection.
HasIDNo: (Has ID No.) The consumer has an ID number. This is similar to a Social Security number in the US and can be useful in legal proceedings. A consumer without any ID No. details is a lot harder to collect debt from. In addition, this field denotes that the account is held by a person and not a business. However, it is not very reliable as it's often not captured properly or at all.
BadDebtIndic: (Bad Debt Indicator) 1 = Is considered to be a Bad Debt, 0 = Not considered to be a Bad Debt
Inspiration
I welcome any feedback on the dataset as well as my methodology in classifying and modelling this dataset. The kernel that I have run against this dataset is my first and I am now working on a second attempt with different parameters. Any advice, criticisms etc - will be much appreciated
====
Target Variable: baddebt (numeric, 2 distinct): ['0', '1']
====
Features:

accountcategoryid (numeric, 12 distinct): ['1', '4', '11', '2', '6', '5', '7', '12', '3', '8']
accountcategory (string, 12 distinct): ['Residential', 'Agricultural', 'Unknown', 'Business', 'Government', 'Municipal', 'Educational', 'Place of Worship', 'Industry', 'Infrastructure']
acccatabbr (string, 12 distinct): ['RES', 'AGR', 'UKN', 'BUS', 'GOV', 'MUN', 'EDU', 'POW', 'IND', 'INF']
propertyvalue (numeric, 11951 distinct): ['0', '60000', '40000', '50000', '35000', '38000', '55000', '1100000', '80000', '34000']
propertysize (numeric, 17424 distinct): ['0', '240', '300', '200', '275', '250', '264', '600', '230', '299']
totalbilling (numeric, 33209 distinct): ['0', '1', '5', '832', '2', '15', '12', '7321', '1079', '2994']
avgbilling (numeric, 6467 distinct): ['0', '1', '2', '3', '4', '5', '8', '7', '83', '6']
totalreceipting (numeric, 27765 distinct): ['0', '100', '200', '50', '500', '1000', '300', '400', '150', '600']
avgreceipting (numeric, 10536 distinct): ['0', '100', '50', '200', '500', '300', '150', '1000', '400', '250']
total90debt (numeric, 25001 distinct): ['0', '1', '1415', '100', '438', '3', '2584', '2', '6', '4']
totalwriteoff (numeric, 6129 distinct): ['0', '2', '155', '233', '243', '181', '13', '7', '376', '194']
collectionratio (numeric, 1967 distinct): ['0.0', '1.0', '0.99', '1.01', '1.02', '0.98', '1.03', '1.04', '1.05', '1.06']
debtbillingratio (numeric, 7467 distinct): ['0.0', '0.01', '-1.0', '1.0', '1.09', '0.02', '1.08', '0.08', '1.07', '0.66']
totalelecbill (numeric, 13717 distinct): ['0', '2126', '703', '3132', '2445', '352', '176', '273', '88', '527']
hasidno (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Municipal Debt Risk Analysis for South African Accounts"
TARGET = CuratedTarget(raw_name='baddebt', new_name='Account Bad Debt', task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'No', '1': 'Yes'})
COLS_TO_DROP = ['accountcategoryid', 'acccatabbr']
FEATURES = [CuratedFeature(raw_name='accountcategory', new_name='Account Category'),
            CuratedFeature(raw_name='propertyvalue', new_name='Property Value'),
            CuratedFeature(raw_name='propertysize', new_name='Property Size'),
            CuratedFeature(raw_name='totalbilling', new_name='Total Billing'),
            CuratedFeature(raw_name='avgbilling', new_name='Average Billing'),
            CuratedFeature(raw_name='totalreceipting', new_name='Total Receipting'),
            CuratedFeature(raw_name='avgreceipting', new_name='Average Receipting'),
            CuratedFeature(raw_name='total90debt', new_name='Total 90+ Debt'),
            CuratedFeature(raw_name='totalwriteoff', new_name='Total Write Off'),
            CuratedFeature(raw_name='collectionratio', new_name='Collection Ratio'),
            CuratedFeature(raw_name='debtbillingratio', new_name='Debt Billing Ratio'),
            CuratedFeature(raw_name='totalelecbill', new_name='Total Electricity Bill'),
            CuratedFeature(raw_name='hasidno', new_name='Has ID Number')
            ]

