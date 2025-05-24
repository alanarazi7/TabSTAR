from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: bank-marketing
====
Examples: 45211
====
URL: https://www.openml.org/search?type=data&id=1461
====
Description: **Author**: Paulo Cortez, Sérgio Moro
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
**Please cite**: S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.       

**Bank Marketing**  
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. 

The classification goal is to predict if the client will subscribe a term deposit (variable y).

### Attribute information  
For more information, read [Moro et al., 2011].

Input variables:

- bank client data:

1 - age (numeric) 

2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur", "student","blue-collar","self-employed","retired","technician","services") 

3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced"  means divorced or widowed) 

4 - education (categorical: "unknown","secondary","primary","tertiary") 

5 - default: has credit in default? (binary: "yes","no") 

6 - balance: average yearly balance, in euros (numeric) 

7 - housing: has housing loan? (binary: "yes","no") 

8 - loan: has personal loan? (binary: "yes","no")

- related with the last contact of the current campaign:

9 - contact: contact communication type (categorical: "unknown","telephone","cellular")

10 - day: last contact day of the month (numeric)

11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

12 - duration: last contact duration, in seconds (numeric)

- other attributes:

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted) 

15 - previous: number of contacts performed before this campaign and for this client (numeric) 

16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
 
- output variable (desired target):

17 - y - has the client subscribed a term deposit? (binary: "yes","no")
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 77 distinct): ['32', '31', '33', '34', '35', '36', '30', '37', '39', '38']
V2 (nominal, 12 distinct): ['blue-collar', 'management', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid']
V3 (nominal, 3 distinct): ['married', 'single', 'divorced']
V4 (nominal, 4 distinct): ['secondary', 'tertiary', 'primary', 'unknown']
V5 (nominal, 2 distinct): ['no', 'yes']
V6 (numeric, 7168 distinct): ['0.0', '1.0', '2.0', '4.0', '3.0', '5.0', '6.0', '8.0', '23.0', '7.0']
V7 (nominal, 2 distinct): ['yes', 'no']
V8 (nominal, 2 distinct): ['no', 'yes']
V9 (nominal, 3 distinct): ['cellular', 'unknown', 'telephone']
V10 (numeric, 31 distinct): ['20', '18', '21', '17', '6', '5', '14', '8', '28', '7']
V11 (nominal, 12 distinct): ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'feb', 'jan', 'oct', 'sep']
V12 (numeric, 1573 distinct): ['124.0', '90.0', '89.0', '104.0', '122.0', '114.0', '136.0', '139.0', '112.0', '121.0']
V13 (numeric, 48 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
V14 (numeric, 559 distinct): ['-1.0', '182.0', '92.0', '91.0', '183.0', '181.0', '370.0', '184.0', '364.0', '95.0']
V15 (numeric, 41 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
V16 (nominal, 4 distinct): ['unknown', 'failure', 'other', 'success']
'''

CONTEXT = "Direct Phone Marketing Campaigns of a Portuguese Banking Institution"
TARGET = CuratedTarget(raw_name="Class", new_name="Subscribed Deposit", task_type=SupervisedTask.BINARY,
                       label_mapping={'1': 'No', '2': 'Yes'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Age"),
            CuratedFeature(raw_name="V2", new_name="Job Profession"),
            CuratedFeature(raw_name="V3", new_name="Marital Status"),
            CuratedFeature(raw_name="V4", new_name="Education Level"),
            CuratedFeature(raw_name="V5", new_name="Has Credit in Default"),
            CuratedFeature(raw_name="V6", new_name="Average Yearly Balance in Euros"),
            CuratedFeature(raw_name="V7", new_name="Has Housing Loan"),
            CuratedFeature(raw_name="V8", new_name="Has Personal Loan"),
            CuratedFeature(raw_name="V9", new_name="Last Contact Communication Type"),
            CuratedFeature(raw_name="V10", new_name="Last Contact Day of the Month"),
            CuratedFeature(raw_name="V11", new_name="Last Contact Month of Year"),
            CuratedFeature(raw_name="V12", new_name="Last Contact Duration in Seconds"),
            CuratedFeature(raw_name="V13", new_name="Number of Contacts Performed During this Campaign"),
            CuratedFeature(raw_name="V14", new_name="Number of Days Passed by After the Client was Last Contacted from a Previous Campaign"),
            CuratedFeature(raw_name="V15", new_name="Number of Contacts Performed Before this Campaign"),
            CuratedFeature(raw_name="V16", new_name="Outcome of the Previous Marketing Campaign"),

    ]
