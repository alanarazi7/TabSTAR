from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: credit-g
====
Examples: 1000
====
URL: https://www.openml.org/search?type=data&id=31
====
Description: **Author**: Dr. Hans Hofmann  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) - 1994    
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

**German Credit dataset**  
This dataset classifies people described by a set of attributes as good or bad credit risks.

This dataset comes with a cost matrix: 
``` 
Good  Bad (predicted)  
Good   0    1   (actual)  
Bad    5    0  
```

It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).  

### Attribute description  

1. Status of existing checking account, in Deutsche Mark.  
2. Duration in months  
3. Credit history (credits taken, paid back duly, delays, critical accounts)  
4. Purpose of the credit (car, television,...)  
5. Credit amount  
6. Status of savings account/bonds, in Deutsche Mark.  
7. Present employment, in number of years.  
8. Installment rate in percentage of disposable income  
9. Personal status (married, single,...) and sex  
10. Other debtors / guarantors  
11. Present residence since X years  
12. Property (e.g. real estate)  
13. Age in years  
14. Other installment plans (banks, stores)  
15. Housing (rent, own,...)  
16. Number of existing credits at this bank  
17. Job  
18. Number of people being liable to provide maintenance for  
19. Telephone (yes,no)  
20. Foreign worker (yes,no)
====
Target Variable: class (nominal, 2 distinct): ['good', 'bad']
====
Features:

checking_status (nominal, 4 distinct): ['no checking', '<0', '0<=X<200', '>=200']
duration (numeric, 33 distinct): ['24', '12', '18', '36', '6', '15', '9', '48', '30', '21']
credit_history (nominal, 5 distinct): ['existing paid', 'critical/other existing credit', 'delayed previously', 'all paid', 'no credits/all paid']
purpose (nominal, 10 distinct): ['radio/tv', 'new car', 'furniture/equipment', 'used car', 'business', 'education', 'repairs', 'domestic appliance', 'other', 'retraining']
credit_amount (numeric, 921 distinct): ['1478.0', '1262.0', '1258.0', '1275.0', '1393.0', '1442.0', '3590.0', '2578.0', '701.0', '1924.0']
savings_status (nominal, 5 distinct): ['<100', 'no known savings', '100<=X<500', '500<=X<1000', '>=1000']
employment (nominal, 5 distinct): ['1<=X<4', '>=7', '4<=X<7', '<1', 'unemployed']
installment_commitment (numeric, 4 distinct): ['4', '2', '3', '1']
personal_status (nominal, 4 distinct): ['male single', 'female div/dep/mar', 'male mar/wid', 'male div/sep', 'female single']
other_parties (nominal, 3 distinct): ['none', 'guarantor', 'co applicant']
residence_since (numeric, 4 distinct): ['4', '2', '3', '1']
property_magnitude (nominal, 4 distinct): ['car', 'real estate', 'life insurance', 'no known property']
age (numeric, 53 distinct): ['27', '26', '23', '24', '28', '25', '30', '35', '36', '31']
other_payment_plans (nominal, 3 distinct): ['none', 'bank', 'stores']
housing (nominal, 3 distinct): ['own', 'rent', 'for free']
existing_credits (numeric, 4 distinct): ['1', '2', '3', '4']
job (nominal, 4 distinct): ['skilled', 'unskilled resident', 'high qualif/self emp/mgmt', 'unemp/unskilled non res']
num_dependents (numeric, 2 distinct): ['1', '2']
own_telephone (nominal, 2 distinct): ['none', 'yes']
foreign_worker (nominal, 2 distinct): ['yes', 'no']
'''


CONTEXT = "Financial credit history of German customers"
TARGET = CuratedTarget(raw_name="class", new_name="German Credit Customer Type", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name="checking_status", new_name="Status of Existing Checking Account in Deutsche Mark",
                   value_mapping={"no checking": "No Checking Account",
                                  "<0": "Negative Balance",
                                  "0<=X<200": "Less than 200 Deutsche Mark",
                                  ">=200": "More than 200 Deutsche Mark"}),
    CuratedFeature(raw_name="duration", new_name="Credit Loan Duration in Months"),
    CuratedFeature(raw_name="savings_status", new_name="Status of savings account/bonds in Deutsche Mark",
                   value_mapping={"<100": "Less than 100 Deutsche Mark",
                                  "no known savings": "No Known Savings",
                                  "100<=X<500": "100 to 500 Deutsche Mark",
                                  "500<=X<1000": "500 to 1000 Deutsche Mark",
                                  ">=1000": "More than 1000 Deutsche Mark"}),
    CuratedFeature(raw_name="employment", new_name="Present Employment Years",
                   value_mapping={"1<=X<4": "1 to 4 Years",
                                  ">=7": "More than 7 Years",
                                  "4<=X<7": "4 to 7 Years",
                                  "<1": "Less than 1 Year",
                                  "unemployed": "Unemployed"}),
    CuratedFeature(raw_name="installment_commitment", new_name="Installment rate in percentage of disposable income",
                   feat_type=FeatureType.NUMERIC),
    CuratedFeature(raw_name="personal_status", new_name="Gender & Personal Marital Status"),
    CuratedFeature(raw_name="other_parties", new_name="Other debtors / guarantors"),
    CuratedFeature(raw_name="residence_since", new_name="Present Residence in Years", feat_type=FeatureType.NUMERIC),
    CuratedFeature(raw_name="property_magnitude", new_name="Property Magnitude"),
    CuratedFeature(raw_name="other_payment_plans", new_name="Other Payment Installment Plans"),
    CuratedFeature(raw_name="housing", new_name="Housing Status"),
    CuratedFeature(raw_name="existing_credits", new_name="Number of Existing Credits at this Bank",
                   feat_type=FeatureType.NUMERIC),
    CuratedFeature(raw_name="job",
                   value_mapping={"high qualif/self emp/mgmt": "Highly Qualified / Self Employed / Management",
                                  "unemp/unskilled non res": "Unemployed / Unskilled Non Resident"}),
]
