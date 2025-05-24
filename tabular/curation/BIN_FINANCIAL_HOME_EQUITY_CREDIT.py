from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: HMEQ_Data
====
Examples: 5960
====
URL: https://www.openml.org/search?type=data&id=43337
====
Description: Context
The consumer credit department of a bank wants to automate the decisionmaking process for approval of home equity lines of credit. To do this, they will follow the recommendations of the Equal Credit Opportunity Act to create an empirically derived and statistically sound credit scoring model. The model will be based on data collected from recent applicants granted credit through the current process of loan underwriting. The model will be built from predictive modeling tools, but the created model must be sufficiently interpretable to provide a reason for any adverse actions (rejections). 
Content
The Home Equity dataset (HMEQ) contains baseline and loan performance information for 5,960 recent home equity loans. The target (BAD) is a binary variable indicating whether an applicant eventually defaulted or was seriously delinquent. This adverse outcome occurred in 1,189 cases (20). For each applicant, 12 input variables were recorded.
Acknowledgements
Inspiration
What if you can predict clients who default on their loans.
====
Target Variable: BAD (numeric, 2 distinct): ['0', '1']
====
Features:

LOAN (numeric, 540 distinct): ['15000', '10000', '20000', '25000', '12000', '17000', '13000', '5000', '11000', '8000']
MORTDUE (numeric, 5054 distinct): ['42000.0', '47000.0', '65000.0', '50000.0', '124000.0', '62000.0', '55000.0', '70000.0', '45000.0', '54000.0']
VALUE (numeric, 5382 distinct): ['60000.0', '80000.0', '85000.0', '65000.0', '78000.0', '72000.0', '50000.0', '87000.0', '68000.0', '83000.0']
REASON (string, 3 distinct): ['DebtCon', 'HomeImp']
JOB (string, 7 distinct): ['Other', 'ProfExe', 'Office', 'Mgr', 'Self', 'Sales']
YOJ (numeric, 100 distinct): ['0.0', '1.0', '2.0', '5.0', '4.0', '6.0', '3.0', '9.0', '10.0', '8.0']
DEROG (numeric, 12 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
DELINQ (numeric, 15 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '10.0']
CLAGE (numeric, 5315 distinct): ['102.5', '206.9667', '177.5', '123.7667', '95.3667', '109.5667', '117.6667', '219.1333', '189.7', '97.4']
NINQ (numeric, 17 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '10.0', '8.0']
CLNO (numeric, 63 distinct): ['16.0', '19.0', '24.0', '23.0', '21.0', '20.0', '18.0', '25.0', '15.0', '22.0']
DEBTINC (numeric, 4694 distinct): ['37.1136', '44.3826', '31.6147', '41.5767', '41.3955', '20.6887', '35.9821', '37.8272', '34.8855', '41.8249']
'''

CONTEXT = "Home Equity Line of Credit Approval Applications"
TARGET = CuratedTarget(raw_name='BAD', new_name='Home Equity Loan Default Status', task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'No Default', '1': 'Default'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name='MORTDUE', new_name='Mortgage Due Amount'),
            CuratedFeature(raw_name='REASON', new_name='Loan Reason',
                           value_mapping={'DebtCon': 'Debt Consolidation', 'HomeImp': 'Home Improvement'}),
            CuratedFeature(raw_name='JOB', new_name='Job Title',
                           value_mapping={'Other': 'Other',
                                          'ProfExe': 'Professional Executive',
                                          'Office': 'Office',
                                          'Mgr': 'Manager',
                                          'Self': 'Self-Employed',
                                          'Sales': 'Sales'}),
            CuratedFeature(raw_name='YOJ', new_name='Years at Job'),
            CuratedFeature(raw_name='DEROG', new_name='Derogatory Reports'),
            CuratedFeature(raw_name='DELINQ', new_name='Delinquent Reports'),
            CuratedFeature(raw_name='CLAGE', new_name='Credit Line Age'),
            CuratedFeature(raw_name='NINQ', new_name='Inquiries in Past 6 Months'),
            CuratedFeature(raw_name='CLNO', new_name='Number of Credit Lines'),
            CuratedFeature(raw_name='DEBTINC', new_name='Debt to Income Ratio')
            ]
