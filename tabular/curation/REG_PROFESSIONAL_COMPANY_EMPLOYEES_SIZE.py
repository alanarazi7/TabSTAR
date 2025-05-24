from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: peopledatalabssf/free-7-million-company-dataset/companies_sorted.csv
====
Examples: 7173426
====
URL: https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset/companies_sorted.csv
====
Description: 
Information on companies with over $1,000$ employees. The task is to predict the number of employees of each company.

====
Features:

Unnamed: 0 (int64, 7173426 distinct): ['5872184', '361379', '361496', '361493', '361399', '361395', '361394', '361393', '361391', '361390']
name (object, 7004634 distinct): ['independent consultant', 'consultant', 'private practice', 'independent', 'independent contractor', 'indépendant', 'confidential', 'autónomo', 'private', 'independiente']
domain (object, 5474764 distinct): ['nordalps.com', 'dunked.com', 'virtualcu.net', 'play-cricket.com', 'notjusttravel.com', 'ypsitesmart.com.au', 'pradocapitalgroup.com', 'corporateonline.in', 'creditunionaccess.com', 'lausd.k12.ca.us']
year founded (float64, 234 distinct): ['2015.0', '2014.0', '2013.0', '2012.0', '2010.0', '2016.0', '2011.0', '2009.0', '2017.0', '2008.0']
industry (object, 148 distinct): ['information technology and services', 'marketing and advertising', 'construction', 'management consulting', 'real estate', 'computer software', 'internet', 'financial services', 'health, wellness and fitness', 'hospital & health care']
size range (object, 8 distinct): ['1 - 10', '11 - 50', '51 - 200', '201 - 500', '501 - 1000', '1001 - 5000', '5001 - 10000', '10001+']
locality (object, 96244 distinct): ['london, greater london, united kingdom', 'new york, new york, united states', 'madrid, madrid, spain', 'paris, île-de-france, france', 'london, london, united kingdom', 'los angeles, california, united states', 'chicago, illinois, united states', 'houston, texas, united states', 'san francisco, california, united states', 'toronto, ontario, canada']
country (object, 236 distinct): ['united states', 'united kingdom', 'canada', 'india', 'spain', 'netherlands', 'germany', 'australia', 'france', 'italy']
linkedin url (object, 7173426 distinct): ['linkedin.com/company/ibm', 'linkedin.com/company/hungarian-table-tennis-association', 'linkedin.com/company/femme-in-stem', "linkedin.com/company/pine's-art-gallery", 'linkedin.com/company/tiwiworld', 'linkedin.com/company/công-ty-cổ-phần-truyền-thông-kim-cương', 'linkedin.com/company/absolute-value-va', 'linkedin.com/company/interlink-logistics-llc', 'linkedin.com/company/awema-ochs-kg', 'linkedin.com/company/sas---security-and-services']
current employee estimate (int64, 5379 distinct): ['1', '0', '2', '3', '4', '5', '6', '7', '8', '9']
total employee estimate (int64, 8486 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
'''

CONTEXT = "Company size prediction"
TARGET = CuratedTarget(raw_name="total employee estimate", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Unnamed: 0", "size range", "current employee estimate"]
FEATURES = []

DESCRIPTION = '''
Information on companies with over $1,000$ employees. The task is to predict the number of employees of each company.
'''