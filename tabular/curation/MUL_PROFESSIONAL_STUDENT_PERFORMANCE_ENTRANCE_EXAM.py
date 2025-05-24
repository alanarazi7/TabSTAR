from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Student_Performance_on_an_Entrance_Examination
====
Examples: 666
====
URL: https://www.openml.org/search?type=data&id=46584
====
Description: Performance in Common Entrance Examination (CEE), Sex of the Candidate, Caste of the Candidate, Whether the candidate attended any coaching classes within Assam, outside Assam or not, Name of the board where the candidate studied at Class X level, Name of the board where the candidate studied at Class XII level, Medium of instructions for the study at Class XII level,The percentage secured by the candidate at Class X standard, The percentage secured by the candidate at Class XII standard, The occupation of the father of the candidate, The occupation of the mother of the candidate
====
Target Variable: Performance (string, 4 distinct): ['Good', 'Vg', 'Average', 'Excellent']
====
Features:

Gender (string, 2 distinct): ['male', 'female']
Caste (string, 4 distinct): ['General', 'OBC', 'ST', 'SC']
coaching (string, 3 distinct): ['WA', 'NO', 'OA']
Class_ten_education (string, 3 distinct): ['SEBA', 'CBSE', 'OTHERS']
twelve_education (string, 3 distinct): ['AHSEC', 'CBSE', 'OTHERS']
medium (string, 3 distinct): ['ENGLISH', 'OTHERS', 'ASSAMESE']
Class_X_Percentage (string, 4 distinct): ['Excellent', 'Vg', 'Good', 'Average']
Class_XII_Percentage (string, 4 distinct): ['Excellent', 'Vg', 'Good', 'Average']
Father_occupation (string, 8 distinct): ['OTHERS', 'SCHOOL_TEACHER', 'BUSINESS', 'DOCTOR', 'ENGINEER', 'COLLEGE_TEACHER', 'CULTIVATOR', 'BANK_OFFICIAL']
Mother_occupation (string, 9 distinct): ['HOUSE_WIFE', 'SCHOOL_TEACHER', 'OTHERS', 'COLLEGE_TEACHER', 'DOCTOR', 'BANK_OFFICIAL', 'BUSINESS', 'ENGINEER', 'CULTIVATOR']
time (string, 6 distinct): ['TWO', 'ONE', 'THREE', 'FOUR', 'FIVE', 'SEVEN']
'''

CONTEXT = "Student Performance on an Entrance Examination"
TARGET = CuratedTarget(raw_name="Performance", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []