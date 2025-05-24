from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: student_performance_por
====
Examples: 649
====
URL: https://www.openml.org/search?type=data&id=44967
====
Description: **Data Description**

This data approach student achievement in secondary education of two Portuguese schools.
The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires.

There are two datasets in the original database, regarding the performance in two distinct subjects: Mathematics and Portuguese language.
This version of the original dataset contains only the latter.

Note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful.

**Attribute Description**

1. *school* - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2. *sex* - student's sex (binary: 'F' - female or 'M' - male)
3. *age* - student's age (numeric: from 15 to 22)
4. *address* - student's home address type (binary: 'U' - urban or 'R' - rural)
5. *famsize* - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
6. *Pstatus* - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
7. *Medu* - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
8. *Fedu* - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
9. *Mjob* - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10. *Fjob* - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11. *reason* - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12. *guardian* - student's guardian (nominal: 'mother', 'father' or 'other')
13. *traveltime* - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14. *studytime* - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15. *failures* - number of past class failures (numeric: n if 1<=n<3, else 4)
16. *schoolsup* - extra educational support (binary: yes or no)
17. *famsup* - family educational support (binary: yes or no)
18. *paid* - extra paid classes within the course subject (binary: yes or no)
19. *activities* - extra-curricular activities (binary: yes or no)
20. *nursery* - attended nursery school (binary: yes or no)
21. *higher* - wants to take higher education (binary: yes or no)
22. *internet* - Internet access at home (binary: yes or no)
23. *romantic* - with a romantic relationship (binary: yes or no)
24. *famrel* - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25. *freetime* - free time after school (numeric: from 1 - very low to 5 - very high)
26. *goout* - going out with friends (numeric: from 1 - very low to 5 - very high)
27. *Dalc* - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28. *Walc* - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
29. *health* - current health status (numeric: from 1 - very bad to 5 - very good)
30. *absences* - number of school absences (numeric)
31. *G1* - first period grade (numeric: from 0 to 20)
32. *G2* - second period grade (numeric: from 0 to 20)
33. *G3* - final grade (numeric: from 0 to 20, target feature)
====
Target Variable: G3 (numeric, 17 distinct): ['11', '10', '13', '12', '14', '15', '16', '9', '8', '17']
====
Features:

school (nominal, 2 distinct): ['GP', 'MS']
sex (nominal, 2 distinct): ['F', 'M']
age (numeric, 8 distinct): ['17', '16', '18', '15', '19', '20', '21', '22']
address (nominal, 2 distinct): ['U', 'R']
famsize (nominal, 2 distinct): ['GT3', 'LE3']
Pstatus (nominal, 2 distinct): ['T', 'A']
Medu (numeric, 5 distinct): ['2', '4', '1', '3', '0']
Fedu (numeric, 5 distinct): ['2', '1', '3', '4', '0']
Mjob (nominal, 5 distinct): ['other', 'services', 'at_home', 'teacher', 'health']
Fjob (nominal, 5 distinct): ['other', 'services', 'at_home', 'teacher', 'health']
reason (nominal, 4 distinct): ['course', 'home', 'reputation', 'other']
guardian (nominal, 3 distinct): ['mother', 'father', 'other']
traveltime (numeric, 4 distinct): ['1', '2', '3', '4']
studytime (numeric, 4 distinct): ['2', '1', '3', '4']
failures (numeric, 4 distinct): ['0', '1', '2', '3']
schoolsup (nominal, 2 distinct): ['no', 'yes']
famsup (nominal, 2 distinct): ['yes', 'no']
paid (nominal, 2 distinct): ['no', 'yes']
activities (nominal, 2 distinct): ['no', 'yes']
nursery (nominal, 2 distinct): ['yes', 'no']
higher (nominal, 2 distinct): ['yes', 'no']
internet (nominal, 2 distinct): ['yes', 'no']
romantic (nominal, 2 distinct): ['no', 'yes']
famrel (numeric, 5 distinct): ['4', '5', '3', '2', '1']
freetime (numeric, 5 distinct): ['3', '4', '2', '5', '1']
goout (numeric, 5 distinct): ['3', '2', '4', '5', '1']
Dalc (numeric, 5 distinct): ['1', '2', '3', '5', '4']
Walc (numeric, 5 distinct): ['1', '2', '3', '4', '5']
health (numeric, 5 distinct): ['5', '3', '4', '1', '2']
absences (numeric, 24 distinct): ['0', '2', '4', '6', '8', '10', '1', '12', '5', '16']
'''

EDU_MAP = {'2': 'Primary Education (4th Grade)', '4': 'Higher Education', '1': '5th to 9th Grade', '3': 'Secondary Education', '0': 'None'}
JOB_MAP = {'other': 'Other', 'services': 'Civil Services', 'at_home': 'At Home', 'teacher': 'Teacher', 'health': 'Health'}

CONTEXT = "Portugal Student Performance"
TARGET = CuratedTarget(raw_name="G3", new_name="Final Grade", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name="school", value_mapping={'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'}),
            CuratedFeature(raw_name="address", value_mapping={'U': 'Urban', 'R': 'Rural'}),
            CuratedFeature(raw_name="famsize", new_name="Family Size",
                           value_mapping={'GT3': 'Greater than 3', 'LE3': 'Less or equal to 3'}),
            CuratedFeature(raw_name="Pstatus", new_name="Parent's Cohabitation Status",
                           value_mapping={'T': 'Living Together', 'A': 'Apart'}),
            CuratedFeature(raw_name="Medu", new_name="Mother's Education", value_mapping=EDU_MAP),
            CuratedFeature(raw_name="Fedu", new_name="Father's Education", value_mapping=EDU_MAP),
            CuratedFeature(raw_name="Mjob", new_name="Mother's Job", value_mapping=JOB_MAP),
            CuratedFeature(raw_name="Fjob", new_name="Father's Job", value_mapping=JOB_MAP),
            CuratedFeature(raw_name="reason", new_name="Reason to Choose School"),
            CuratedFeature(raw_name="traveltime", new_name="Travel Time",
                           value_mapping={'1': 'Less than 15 minutes', '2': '15 to 30 minutes',
                                          '3': '30 minutes to 1 hour', '4': 'More than 1 hour'}),
            CuratedFeature(raw_name="studytime", new_name="Study Time",
                            value_mapping={'1': 'Less than 2 hours', '2': '2 to 5 hours',
                                            '3': '5 to 10 hours', '4': 'More than 10 hours'}),
            CuratedFeature(raw_name="failures", new_name="Past Class Failures"),
            CuratedFeature(raw_name="schoolsup", new_name="Extra Educational Support"),
            CuratedFeature(raw_name="famsup", new_name="Family Educational Support"),
            CuratedFeature(raw_name="paid", new_name="Extra Paid Classes"),
            CuratedFeature(raw_name="activities", new_name="Extra-curricular Activities"),
            CuratedFeature(raw_name="nursery", new_name="Attended Nursery School"),
            CuratedFeature(raw_name="higher", new_name="Wants Higher Education"),
            CuratedFeature(raw_name="internet", new_name="Internet Access at Home"),
            CuratedFeature(raw_name="romantic", new_name="With Romantic Relationship"),
            CuratedFeature(raw_name="famrel", new_name="Quality of Family Relationships",
                           value_mapping={'1': 'Very Bad', '2': 'Bad', '3': 'Neutral', '4': 'Good', '5': 'Excellent'}),
            CuratedFeature(raw_name="freetime", new_name="Free Time After School",
                           value_mapping={'1': 'Very Low', '2': 'Low', '3': 'Neutral', '4': 'High', '5': 'Very High'}),
            CuratedFeature(raw_name="goout", new_name="Going Out with Friends",
                           value_mapping={'1': 'Very Low', '2': 'Low', '3': 'Neutral', '4': 'High', '5': 'Very High'}),
            CuratedFeature(raw_name="Dalc", new_name="Workday Alcohol Consumption",
                           value_mapping={'1': 'Very Low', '2': 'Low', '3': 'Neutral', '4': 'High', '5': 'Very High'}),
            CuratedFeature(raw_name="Walc", new_name="Weekend Alcohol Consumption",
                           value_mapping={'1': 'Very Low', '2': 'Low', '3': 'Neutral', '4': 'High', '5': 'Very High'}),
            CuratedFeature(raw_name="health", new_name="Current Health Status",
                           value_mapping={'1': 'Very Bad', '2': 'Bad', '3': 'Neutral', '4': 'Good', '5': 'Very Good'}),
            CuratedFeature(raw_name="absences", new_name="Number of School Absences")]

