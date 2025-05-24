from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT
====
Examples: 29165
====
URL: https://www.scimagojr.com/journalrank.php?out=xls
====
Description: 
Scientific journals and their descriptive features from Scimago journal rank. The task is to predict the H-index of journals.

====
Features:

Rank (int64, 29165 distinct): ['1', '19426', '19452', '19451', '19450', '19449', '19448', '19447', '19446', '19445']
Sourceid (int64, 29165 distinct): ['28773', '17600155052', '19900191813', '16200154763', '21100780829', '21100825345', '21100896480', '21101168860', '21100463178', '21100802779']
Title (object, 29144 distinct): ['Agenda', 'Journal of Marine Science and Technology', 'Surgery', 'Recherche et Applications en Marketing', 'Engineering', 'Public Policy and Administration', 'Philosophical Magazine', 'Investigaciones Geograficas', 'Environmental Chemistry', 'Portal']
Type (object, 4 distinct): ['journal', 'book series', 'conference and proceedings', 'trade journal']
Issn (object, 29137 distinct): ['-', '16088751, 25671014', '15424863, 00079235', '09328114', '17263247', '19945124, 24112658', '21114838, 19563485', '26587149, 26584670', '1646043X, 21829942', '24508187, 24507458']
SJR (object, 2905 distinct): ['0,101', '0,100', '0,102', '0,133', '0,103', '0,111', '0,104', '0,116', '0,105', '0,110']
SJR Best Quartile (object, 5 distinct): ['Q1', 'Q2', 'Q3', 'Q4', '-']
H index (int64, 430 distinct): ['4', '5', '3', '6', '7', '8', '2', '9', '10', '11']
Total Docs. (2023) (int64, 1206 distinct): ['0', '20', '24', '16', '22', '18', '21', '17', '14', '15']
Total Docs. (3years) (int64, 2228 distinct): ['61', '66', '84', '58', '47', '60', '57', '74', '75', '70']
Total Refs. (int64, 9607 distinct): ['0', '1207', '950', '922', '666', '1123', '588', '632', '700', '196']
Total Cites (3years) (int64, 4366 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '8', '10']
Citable Docs. (3years) (int64, 2110 distinct): ['1', '57', '64', '60', '56', '59', '70', '37', '69', '67']
Cites / Doc. (2years) (object, 1341 distinct): ['0,00', '0,07', '0,13', '0,10', '0,11', '0,04', '0,06', '0,08', '0,05', '0,03']
Ref. / Doc. (object, 8053 distinct): ['0,00', '30,00', '46,00', '36,00', '35,00', '29,00', '44,00', '34,00', '48,00', '43,00']
%Female (object, 5473 distinct): ['0,00', '50,00', '33,33', '25,00', '40,00', '20,00', '42,86', '28,57', '66,67', '37,50']
Overton (int64, 63 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SDG (int64, 657 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Country (object, 116 distinct): ['United States', 'United Kingdom', 'Netherlands', 'Germany', 'China', 'Switzerland', 'Spain', 'Italy', 'Poland', 'Russian Federation']
Region (object, 9 distinct): ['Western Europe', 'Northern America', 'Asiatic Region', 'Eastern Europe', 'Latin America', 'Middle East', 'Pacific Region', 'Africa', 'Africa/Middle East']
Publisher (object, 8175 distinct): ['Taylor and Francis Ltd.', 'Elsevier B.V.', 'Routledge', 'SAGE Publications Inc.', 'Wiley-Blackwell Publishing Ltd', 'Elsevier Ltd', 'Oxford University Press', 'SAGE Publications Ltd', 'Brill Academic Publishers', 'Springer Netherlands']
Coverage (object, 5475 distinct): ['2019-2023', '2018-2023', '2008-2023', '2009-2023', '2010-2023', '2011-2023', '2017-2023', '1996-2023', '2013-2023', '2015-2023']
Categories (object, 15935 distinct): ['Medicine (miscellaneous) (Q4)', 'Medicine (miscellaneous) (Q3)', 'Linguistics and Language (Q2)', 'Education (Q1)', 'Education (Q3)', 'Law (Q4)', 'Education (Q2)', 'Literature and Literary Theory (Q4)', 'Law (Q3)', 'Linguistics and Language (Q3)']
Areas (object, 1223 distinct): ['Medicine', 'Social Sciences', 'Arts and Humanities; Social Sciences', 'Arts and Humanities', 'Agricultural and Biological Sciences', 'Mathematics', 'Engineering', 'Biochemistry, Genetics and Molecular Biology; Medicine', 'Earth and Planetary Sciences', 'Computer Science']
'''

CONTEXT = "Academic impact for Scientific Journals"
TARGET = CuratedTarget(raw_name="H index", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Sourceid"]
FEATURES = []

DESCRIPTION = '''
Scientific journals and their descriptive features from Scimago journal rank. The task is to predict the H-index of journals.
'''