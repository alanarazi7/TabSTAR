from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Loan-Predication
====
Examples: 614
====
URL: https://www.openml.org/search?type=data&id=43595
====
Description: Among all industries, insurance domain has the largest use of analytics  data science methods. This data set would provide you enough taste of working on data sets from insurance companies, what challenges are faced, what strategies are used, which variables influence the outcome etc. This is a classification problem. The data has 615 rows and 13 columns.
Problem-----
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.
====
Target Variable: Loan_Status (string, 2 distinct): ['Y', 'N']
====
Features:

Gender (string, 3 distinct): ['Male', 'Female']
Married (string, 3 distinct): ['Yes', 'No']
Dependents (string, 5 distinct): ['0', '1', '2', '3+']
Education (string, 2 distinct): ['Graduate', 'Not Graduate']
Self_Employed (string, 3 distinct): ['No', 'Yes']
ApplicantIncome (numeric, 505 distinct): ['2500', '4583', '6000', '2600', '3333', '4166', '3750', '5000', '8333', '6250']
CoapplicantIncome (numeric, 287 distinct): ['0.0', '2500.0', '2083.0', '1666.0', '2250.0', '1750.0', '1800.0', '1625.0', '2333.0', '1459.0']
LoanAmount (numeric, 204 distinct): ['120.0', '110.0', '100.0', '160.0', '187.0', '128.0', '113.0', '130.0', '95.0', '96.0']
Loan_Amount_Term (numeric, 11 distinct): ['360.0', '180.0', '480.0', '300.0', '240.0', '84.0', '120.0', '60.0', '36.0', '12.0']
Credit_History (numeric, 3 distinct): ['1.0', '0.0']
Property_Area (string, 3 distinct): ['Semiurban', 'Urban', 'Rural']
'''

CONTEXT = "Customers evaluated for loan eligibility process."
TARGET = CuratedTarget(raw_name="Loan_Status", new_name="Loan Eligibility Status", task_type=SupervisedTask.BINARY,
                       label_mapping={"Y": "Yes", "N": "No"})
COLS_TO_DROP = ["Loan_ID"]
FEATURES = [
    CuratedFeature(raw_name="Dependents", new_name="Number of Dependent People"),
    CuratedFeature(raw_name="Education", new_name="Education Status"),
    CuratedFeature(raw_name="ApplicantIncome", new_name="Applicant Income"),
    CuratedFeature(raw_name="CoapplicantIncome", new_name="Coapplicant Income"),
    CuratedFeature(raw_name="LoanAmount", new_name="Loan Amount"),
    CuratedFeature(raw_name="Loan_Amount_Term", new_name="Loan Amount Period Term in Months"),
    CuratedFeature(raw_name="Credit_History", new_name="Credit History Type",
                   value_mapping={"1.0": "Positive", "0.0": "Negative"}),
            ]
