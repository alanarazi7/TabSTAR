from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: employee_salaries
====
Examples: 9228
====
URL: https://www.openml.org/search?type=data&id=42125
====
Description: Annual salary information including gross pay and overtime pay for all active, permanent employees of Montgomery County, MD paid in calendar year 2016. This information will be published annually each year.
====
Target Variable: current_annual_salary (numeric, 3403 distinct): ['92756.7', '89620.0', '102664.0', '69222.18', '73801.0', '100849.36', '93396.0', '45261.0', '97912.0', '50910.0']
====
Features:

full_name (string, 9222 distinct): ['Wong, Ka Y.', 'Miller, Michael E.', 'Cruz, Angela', 'Carter, Jerome', 'Smith, Beverly E.', 'Smith, Jason M.', 'Niblock, David K.', 'Nice, Matthew L.', 'Nicholas, Jamald F.', 'Nicholas, Jerome A.']
gender (nominal, 2 distinct): ['M', 'F']
2016_gross_pay_received (numeric, 8977 distinct): ['119244.9', '625.0', '120825.94', '119244.91', '101542.68', '103397.41', '528.0', '101542.35', '0.0', '98611.71']
2016_overtime_pay (numeric, 6176 distinct): ['0.01', '73.64', '57.87', '0.0', '94.51', '64.17', '68.73', '66.41', '36.21', '87.03']
department (nominal, 37 distinct): ['POL', 'HHS', 'FRS', 'DOT', 'COR', 'DLC', 'DGS', 'LIB', 'DPS', 'SHF']
department_name (nominal, 37 distinct): ['Department of Police', 'Department of Health and Human Services', 'Fire and Rescue Services', 'Department of Transportation', 'Correction and Rehabilitation', 'Department of Liquor Control', 'Department of General Services', 'Department of Public Libraries', 'Department of Permitting Services', "Sheriff's Office"]
division (string, 694 distinct): ['School Health Services', 'Transit Silver Spring Ride On', 'Transit Gaithersburg Ride On', 'Highway Services', 'Child Welfare Services', 'FSB Traffic Division School Safety Section', 'Income Supports', 'PSB 3rd District Patrol', 'PSB 4th District Patrol', 'Transit Nicholson Ride On']
assignment_category (nominal, 2 distinct): ['Fulltime-Regular', 'Parttime-Regular']
employee_position_title (string, 385 distinct): ['Police Officer III', 'Firefighter/Rescuer III', 'Bus Operator', 'Manager III', 'Correctional Officer III (Corporal)', 'Master Firefighter/Rescuer', 'Office Services Coordinator', 'School Health Room Technician I', 'Community Health Nurse II', 'Crossing Guard']
underfilled_job_title (string, 84 distinct): ['Firefighter/Rescuer II', 'Police Officer II', 'Police Officer I', 'Firefighter/Rescuer I (Recruit)', 'Correctional Officer II (PFC)', 'Public Safety Communications Specialist I', 'Public Safety Communications Specialist II', 'Supply Technician II', 'Permitting and Code Enforcement Inspector II', 'Permitting Services Specialist II']
date_first_hired (string, 2264 distinct): ['12/12/2016', '01/14/2013', '02/24/2014', '03/10/2014', '08/12/2013', '10/06/2014', '09/22/2014', '03/19/2007', '07/29/2013', '07/16/2012']
year_first_hired (numeric, 51 distinct): ['2014', '2013', '2016', '2006', '2007', '2012', '2008', '2015', '2001', '2002']
'''

CONTEXT = "Employee Salary in Montgomery County, MD"
TARGET = CuratedTarget(raw_name="current_annual_salary", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["year_first_hired"]
FEATURES = [CuratedFeature(raw_name="date_first_hired", feat_type=FeatureType.DATE)]