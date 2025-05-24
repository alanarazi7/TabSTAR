from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: medical_charges
====
Examples: 163065
====
URL: https://www.openml.org/search?type=data&id=44146
====
Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark, transformed in the same way. This dataset belongs to the "regression on numerical features" benchmark. Original description: 
 
The Inpatient Utilization and Payment Public Use File (Inpatient PUF) provides information on inpatient discharges for Medicare fee-for-service beneficiaries. The Inpatient PUF includes information on utilization, payment (total payment and Medicare payment), and hospital-specific charges for the more than 3,000 U.S. hospitals that receive Medicare Inpatient Prospective Payment System (IPPS) payments. The PUF is organized by hospital and Medicare Severity Diagnosis Related Group (MS-DRG) and covers Fiscal Year (FY) 2011 through FY 2016.
====
Target Variable: AverageTotalPayments (numeric, 154891 distinct): ['8.3827', '8.6079', '8.613', '8.3388', '8.2779', '8.2618', '8.3708', '8.3777', '8.2995', '8.4958']
====
Features:

Total_Discharges (numeric, 642 distinct): ['11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0', '18.0', '19.0', '20.0']
Average_Covered_Charges (numeric, 161985 distinct): ['31155.0', '27512.0', '16015.0', '22990.25', '16723.0', '25073.0', '22858.0', '38805.0', '19144.0', '17402.0']
Average_Medicare_Payments (numeric, 157817 distinct): ['3862.0', '3744.0', '4908.0', '3358.0', '5035.0', '4173.0', '3556.0', '5956.0', '3512.0', '3089.0']
'''

CONTEXT = "Medical Charges"
TARGET = CuratedTarget(raw_name="AverageTotalPayments", new_name="Average Total Payments",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []