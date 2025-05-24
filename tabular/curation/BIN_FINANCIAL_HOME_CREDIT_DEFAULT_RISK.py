from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: hcdr_main
====
Examples: 307511
====
URL: https://www.openml.org/search?type=data&id=45567
====
Description: Home Credit Default Risk Main Table

**WARNING:** This is only the main table of the competition' training dataset! Please do not use it alone (but rather use all data available on Kaggle) unless you aim to reproduce the results of:

> Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). 
> Tabtransformer: Tabular data modeling using contextual embeddings. 
> arXiv preprint arXiv:2012.06678v1.

Check the [Kaggle competition website](https://www.kaggle.com/competitions/home-credit-default-risk) for further information.

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit Group

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.
====
Target Variable: TARGET (nominal, 2 distinct): ['0', '1']
====
Features:

NAME_CONTRACT_TYPE (nominal, 2 distinct): ['Cash loans', 'Revolving loans']
CODE_GENDER (nominal, 3 distinct): ['F', 'M', 'XNA']
FLAG_OWN_CAR (nominal, 2 distinct): ['N', 'Y']
FLAG_OWN_REALTY (nominal, 2 distinct): ['Y', 'N']
CNT_CHILDREN (numeric, 15 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '14', '8']
AMT_INCOME_TOTAL (numeric, 2548 distinct): ['135000.0', '112500.0', '157500.0', '180000.0', '90000.0', '225000.0', '202500.0', '67500.0', '270000.0', '81000.0']
AMT_CREDIT (numeric, 5603 distinct): ['450000.0', '675000.0', '225000.0', '180000.0', '270000.0', '900000.0', '254700.0', '545040.0', '808650.0', '135000.0']
AMT_ANNUITY (numeric, 13673 distinct): ['9000.0', '13500.0', '6750.0', '10125.0', '37800.0', '11250.0', '26217.0', '20250.0', '12375.0', '31653.0']
AMT_GOODS_PRICE (numeric, 1003 distinct): ['450000.0', '225000.0', '675000.0', '900000.0', '270000.0', '180000.0', '454500.0', '1125000.0', '135000.0', '315000.0']
NAME_TYPE_SUITE (nominal, 8 distinct): ['Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']
NAME_INCOME_TYPE (nominal, 8 distinct): ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed', 'Student', 'Businessman', 'Maternity leave']
NAME_EDUCATION_TYPE (nominal, 5 distinct): ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
NAME_FAMILY_STATUS (nominal, 6 distinct): ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow', 'Unknown']
NAME_HOUSING_TYPE (nominal, 6 distinct): ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
REGION_POPULATION_RELATIVE (numeric, 81 distinct): ['0.0358', '0.0462', '0.0308', '0.0252', '0.0264', '0.0313', '0.0287', '0.0191', '0.0725', '0.0207']
DAYS_BIRTH (numeric, 17460 distinct): ['-13749.0', '-13481.0', '-18248.0', '-10020.0', '-15771.0', '-10292.0', '-14267.0', '-13263.0', '-11664.0', '-14395.0']
DAYS_EMPLOYED (numeric, 12574 distinct): ['365243.0', '-200.0', '-224.0', '-230.0', '-199.0', '-212.0', '-384.0', '-229.0', '-231.0', '-215.0']
DAYS_REGISTRATION (numeric, 15688 distinct): ['-1.0', '-7.0', '-6.0', '-4.0', '-2.0', '-5.0', '-3.0', '-9.0', '-14.0', '-21.0']
DAYS_ID_PUBLISH (numeric, 6168 distinct): ['-4053.0', '-4095.0', '-4046.0', '-4417.0', '-4256.0', '-4032.0', '-4151.0', '-4200.0', '-4171.0', '-4214.0']
OWN_CAR_AGE (numeric, 63 distinct): ['7.0', '6.0', '3.0', '8.0', '2.0', '4.0', '1.0', '9.0', '10.0', '14.0']
FLAG_MOBIL (nominal, 2 distinct): ['1', '0']
FLAG_EMP_PHONE (nominal, 2 distinct): ['1', '0']
FLAG_WORK_PHONE (nominal, 2 distinct): ['0', '1']
FLAG_CONT_MOBILE (nominal, 2 distinct): ['1', '0']
FLAG_PHONE (nominal, 2 distinct): ['0', '1']
FLAG_EMAIL (nominal, 2 distinct): ['0', '1']
OCCUPATION_TYPE (nominal, 19 distinct): ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff']
CNT_FAM_MEMBERS (numeric, 18 distinct): ['2.0', '1.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
REGION_RATING_CLIENT (numeric, 3 distinct): ['2', '3', '1']
REGION_RATING_CLIENT_W_CITY (numeric, 3 distinct): ['2', '3', '1']
WEEKDAY_APPR_PROCESS_START (nominal, 7 distinct): ['TUESDAY', 'WEDNESDAY', 'MONDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
HOUR_APPR_PROCESS_START (numeric, 24 distinct): ['10', '11', '12', '13', '14', '9', '15', '16', '8', '17']
REG_REGION_NOT_LIVE_REGION (nominal, 2 distinct): ['0', '1']
REG_REGION_NOT_WORK_REGION (nominal, 2 distinct): ['0', '1']
LIVE_REGION_NOT_WORK_REGION (nominal, 2 distinct): ['0', '1']
REG_CITY_NOT_LIVE_CITY (nominal, 2 distinct): ['0', '1']
REG_CITY_NOT_WORK_CITY (nominal, 2 distinct): ['0', '1']
LIVE_CITY_NOT_WORK_CITY (nominal, 2 distinct): ['0', '1']
ORGANIZATION_TYPE (nominal, 58 distinct): ['Business Entity Type 3', 'XNA', 'Self-employed', 'Other', 'Medicine', 'Business Entity Type 2', 'Government', 'School', 'Trade: type 7', 'Kindergarten']
EXT_SOURCE_1 (numeric, 114585 distinct): ['0.6227', '0.444', '0.5464', '0.581', '0.499', '0.5985', '0.5282', '0.6677', '0.6052', '0.3563']
EXT_SOURCE_2 (numeric, 119832 distinct): ['0.2859', '0.2623', '0.2653', '0.1597', '0.2653', '0.2665', '0.2631', '0.1621', '0.1622', '0.1632']
EXT_SOURCE_3 (numeric, 815 distinct): ['0.7463', '0.7136', '0.6941', '0.6707', '0.6529', '0.5815', '0.6895', '0.5955', '0.5549', '0.6212']
APARTMENTS_AVG (numeric, 2340 distinct): ['0.0825', '0.0619', '0.0928', '0.0722', '0.0082', '0.0165', '0.1031', '0.1485', '0.0124', '0.0742']
BASEMENTAREA_AVG (numeric, 3781 distinct): ['0.0', '0.0545', '0.0818', '0.0727', '0.1091', '0.0796', '0.08', '0.0805', '0.0764', '0.0793']
YEARS_BEGINEXPLUATATION_AVG (numeric, 286 distinct): ['0.9871', '0.9856', '0.9861', '0.9801', '0.9866', '0.9806', '0.9851', '0.9811', '0.9816', '0.9831']
YEARS_BUILD_AVG (numeric, 150 distinct): ['0.8232', '0.8164', '0.8028', '0.728', '0.7348', '0.8096', '0.83', '0.796', '0.7484', '0.7688']
COMMONAREA_AVG (numeric, 3182 distinct): ['0.0', '0.0079', '0.0078', '0.008', '0.0077', '0.0086', '0.0014', '0.007', '0.0013', '0.0069']
ELEVATORS_AVG (numeric, 258 distinct): ['0.0', '0.08', '0.16', '0.24', '0.12', '0.04', '0.2', '0.32', '0.28', '0.4']
ENTRANCES_AVG (numeric, 286 distinct): ['0.1379', '0.069', '0.1034', '0.2069', '0.0345', '0.1724', '0.2759', '0.2414', '0.3448', '0.3103']
FLOORSMAX_AVG (numeric, 404 distinct): ['0.1667', '0.3333', '0.0417', '0.375', '0.125', '0.0833', '0.0', '0.4583', '0.625', '0.5417']
FLOORSMIN_AVG (numeric, 306 distinct): ['0.2083', '0.375', '0.0417', '0.0833', '0.4167', '0.1667', '0.125', '0.0', '0.5', '0.6667']
LANDAREA_AVG (numeric, 3528 distinct): ['0.0', '0.0631', '0.0316', '0.0473', '0.0174', '0.0237', '0.0552', '0.0158', '0.0331', '0.0189']
LIVINGAPARTMENTS_AVG (numeric, 1869 distinct): ['0.0504', '0.0672', '0.0756', '0.0588', '0.0841', '0.121', '0.0067', '0.0605', '0.1009', '0.0134']
LIVINGAREA_AVG (numeric, 5200 distinct): ['0.0', '0.0512', '0.051', '0.0702', '0.0509', '0.0538', '0.0638', '0.0513', '0.0511', '0.0626']
NONLIVINGAPARTMENTS_AVG (numeric, 387 distinct): ['0.0', '0.0039', '0.0077', '0.0116', '0.0154', '0.0193', '0.0019', '0.0232', '0.027', '0.0309']
NONLIVINGAREA_AVG (numeric, 3291 distinct): ['0.0', '0.0012', '0.0044', '0.0022', '0.0031', '0.001', '0.0011', '0.0036', '0.003', '0.0024']
APARTMENTS_MODE (numeric, 761 distinct): ['0.084', '0.063', '0.0945', '0.0735', '0.0084', '0.0168', '0.105', '0.1513', '0.0126', '0.0756']
BASEMENTAREA_MODE (numeric, 3842 distinct): ['0.0', '0.0566', '0.0849', '0.0642', '0.083', '0.1132', '0.0792', '0.0826', '0.0679', '0.0755']
YEARS_BEGINEXPLUATATION_MODE (numeric, 222 distinct): ['0.9871', '0.9866', '0.9861', '0.9801', '0.9806', '0.9856', '0.9851', '0.9816', '0.9796', '0.9791']
YEARS_BUILD_MODE (numeric, 155 distinct): ['0.8301', '0.8236', '0.7387', '0.8171', '0.8105', '0.7452', '0.8367', '0.804', '0.7583', '0.7779']
COMMONAREA_MODE (numeric, 3129 distinct): ['0.0', '0.008', '0.0079', '0.0078', '0.0081', '0.0087', '0.0014', '0.007', '0.0071', '0.0012']
ELEVATORS_MODE (numeric, 27 distinct): ['0.0', '0.0806', '0.1611', '0.2417', '0.1208', '0.0403', '0.2014', '0.3222', '0.282', '0.4028']
ENTRANCES_MODE (numeric, 31 distinct): ['0.1379', '0.069', '0.1034', '0.2069', '0.0345', '0.1724', '0.2759', '0.2414', '0.3448', '0.3103']
FLOORSMAX_MODE (numeric, 26 distinct): ['0.1667', '0.3333', '0.0417', '0.375', '0.125', '0.0833', '0.0', '0.4583', '0.625', '0.5417']
FLOORSMIN_MODE (numeric, 26 distinct): ['0.2083', '0.375', '0.0417', '0.0833', '0.4167', '0.1667', '0.125', '0.0', '0.5', '0.6667']
LANDAREA_MODE (numeric, 3564 distinct): ['0.0', '0.0194', '0.0645', '0.0484', '0.0323', '0.0147', '0.0144', '0.0111', '0.0258', '0.0242']
LIVINGAPARTMENTS_MODE (numeric, 737 distinct): ['0.0551', '0.0735', '0.0826', '0.0643', '0.0918', '0.1322', '0.0661', '0.0073', '0.0147', '0.0588']
LIVINGAREA_MODE (numeric, 5302 distinct): ['0.0', '0.053', '0.0532', '0.0529', '0.0533', '0.0561', '0.0656', '0.0734', '0.0536', '0.0525']
NONLIVINGAPARTMENTS_MODE (numeric, 168 distinct): ['0.0', '0.0039', '0.0078', '0.0117', '0.0156', '0.0195', '0.0233', '0.0272', '0.0311', '0.035']
NONLIVINGAREA_MODE (numeric, 3328 distinct): ['0.0', '0.0011', '0.0046', '0.0033', '0.0012', '0.0023', '0.0013', '0.003', '0.0055', '0.0037']
APARTMENTS_MEDI (numeric, 1149 distinct): ['0.0833', '0.0625', '0.0937', '0.0729', '0.0083', '0.0167', '0.1041', '0.1499', '0.0125', '0.0749']
BASEMENTAREA_MEDI (numeric, 3773 distinct): ['0.0', '0.0818', '0.1091', '0.0545', '0.0727', '0.0796', '0.08', '0.0805', '0.0655', '0.0797']
YEARS_BEGINEXPLUATATION_MEDI (numeric, 246 distinct): ['0.9871', '0.9861', '0.9856', '0.9866', '0.9801', '0.9806', '0.9851', '0.9796', '0.9876', '0.9816']
YEARS_BUILD_MEDI (numeric, 152 distinct): ['0.8256', '0.8189', '0.8054', '0.7316', '0.8121', '0.8323', '0.7383', '0.7987', '0.7518', '0.7719']
COMMONAREA_MEDI (numeric, 3203 distinct): ['0.0', '0.0079', '0.008', '0.0078', '0.0014', '0.0086', '0.0081', '0.0071', '0.0087', '0.0012']
ELEVATORS_MEDI (numeric, 47 distinct): ['0.0', '0.08', '0.16', '0.24', '0.12', '0.04', '0.2', '0.32', '0.28', '0.4']
ENTRANCES_MEDI (numeric, 47 distinct): ['0.1379', '0.069', '0.1034', '0.2069', '0.0345', '0.1724', '0.2759', '0.2414', '0.3448', '0.3103']
FLOORSMAX_MEDI (numeric, 50 distinct): ['0.1667', '0.3333', '0.0417', '0.375', '0.125', '0.0833', '0.4583', '0.0', '0.625', '0.5417']
FLOORSMIN_MEDI (numeric, 48 distinct): ['0.2083', '0.375', '0.0417', '0.0833', '0.4167', '0.1667', '0.125', '0.0', '0.5', '0.6667']
LANDAREA_MEDI (numeric, 3561 distinct): ['0.0', '0.0193', '0.0642', '0.0482', '0.0143', '0.0161', '0.0803', '0.0241', '0.0321', '0.0178']
LIVINGAPARTMENTS_MEDI (numeric, 1098 distinct): ['0.0513', '0.0684', '0.077', '0.0599', '0.0855', '0.1231', '0.0068', '0.0616', '0.1026', '0.0137']
LIVINGAREA_MEDI (numeric, 5282 distinct): ['0.0', '0.0548', '0.0518', '0.052', '0.0521', '0.0522', '0.0513', '0.0519', '0.0888', '0.0717']
NONLIVINGAPARTMENTS_MEDI (numeric, 215 distinct): ['0.0', '0.0039', '0.0078', '0.0116', '0.0155', '0.0194', '0.0019', '0.0233', '0.0272', '0.0311']
NONLIVINGAREA_MEDI (numeric, 3324 distinct): ['0.0', '0.0012', '0.0022', '0.0037', '0.0044', '0.0011', '0.0043', '0.0031', '0.001', '0.0029']
FONDKAPREMONT_MODE (nominal, 5 distinct): ['reg oper account', 'reg oper spec account', 'not specified', 'org spec account']
HOUSETYPE_MODE (nominal, 4 distinct): ['block of flats', 'specific housing', 'terraced house']
TOTALAREA_MODE (numeric, 5117 distinct): ['0.0', '0.057', '0.0547', '0.055', '0.0555', '0.0548', '0.0551', '0.0573', '0.0554', '0.0566']
WALLSMATERIAL_MODE (nominal, 8 distinct): ['Panel', 'Stone, brick', 'Block', 'Wooden', 'Mixed', 'Monolithic', 'Others']
EMERGENCYSTATE_MODE (nominal, 3 distinct): ['No', 'Yes']
OBS_30_CNT_SOCIAL_CIRCLE (numeric, 34 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
DEF_30_CNT_SOCIAL_CIRCLE (numeric, 11 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '34.0', '8.0']
OBS_60_CNT_SOCIAL_CIRCLE (numeric, 34 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
DEF_60_CNT_SOCIAL_CIRCLE (numeric, 10 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '24.0']
DAYS_LAST_PHONE_CHANGE (numeric, 3774 distinct): ['0.0', '-1.0', '-2.0', '-3.0', '-4.0', '-5.0', '-6.0', '-7.0', '-8.0', '-476.0']
FLAG_DOCUMENT_2 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_3 (nominal, 2 distinct): ['1', '0']
FLAG_DOCUMENT_4 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_5 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_6 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_7 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_8 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_9 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_10 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_11 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_12 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_13 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_14 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_15 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_16 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_17 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_18 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_19 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_20 (nominal, 2 distinct): ['0', '1']
FLAG_DOCUMENT_21 (nominal, 2 distinct): ['0', '1']
AMT_REQ_CREDIT_BUREAU_HOUR (numeric, 6 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0']
AMT_REQ_CREDIT_BUREAU_DAY (numeric, 10 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '9.0', '8.0']
AMT_REQ_CREDIT_BUREAU_WEEK (numeric, 10 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '6.0', '5.0', '8.0', '7.0']
AMT_REQ_CREDIT_BUREAU_MON (numeric, 25 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '9.0', '8.0']
AMT_REQ_CREDIT_BUREAU_QRT (numeric, 12 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '7.0', '261.0']
AMT_REQ_CREDIT_BUREAU_YEAR (numeric, 26 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
'''

CONTEXT = "Home Credit Default Risk Prediction"
TARGET = CuratedTarget(raw_name="TARGET", new_name="Credit Loan Status", task_type=SupervisedTask.BINARY,
                       label_mapping={"0": "Non-Default", "1": "Default"})
COLS_TO_DROP = []
FEATURES = []
