from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: Is_fraud
====
Examples: 5227
====
URL: https://www.openml.org/search?type=data&id=46369
====
Description: A fraud detection dataset for binary classification. The target variable is 'is_fraud', indicating whether a transaction is fraudulent.
====
Target Variable: is_fraud (nominal, 2 distinct): ['not_fraudulent', 'fraudulent']
====
Features:

trans_date_trans_time (string, 5135 distinct): ['2020-12-13 17:53:47', '2020-12-12 23:40:18', '2020-12-13 01:48:38', '2020-12-14 10:29:26', '2020-12-14 12:50:13', '2020-12-13 05:18:52', '2020-12-13 21:18:27', '2020-12-13 13:21:52', '2020-12-14 14:51:21', '2020-12-12 22:00:55']
cc_num (numeric, 668 distinct): ['372520049757633.0', '3576431665303017.0', '2242542703101232.8', '6538441737335434.0', '3567697931646329.0', '503874407318.0', '6011109736646996.0', '5540636818935089.0', '3545109339866548.0', '344342339068828.0']
merchant (string, 476 distinct): ['fraud_Conroy-Cruickshank', 'fraud_Kilback LLC', 'fraud_Hackett-Lueilwitz', 'fraud_Schumm PLC', 'fraud_Koss and Sons', 'fraud_Kling Inc', 'fraud_Huels-Hahn', 'fraud_Goyette-Gerhold', 'fraud_Parisian and Sons', "fraud_Friesen-D'Amore"]
category (string, 14 distinct): ['gas_transport', 'grocery_pos', 'kids_pets', 'shopping_pos', 'misc_pos', 'home', 'entertainment', 'shopping_net', 'personal_care', 'health_fitness']
amt (numeric, 4147 distinct): ['3.76', '4.43', '5.4', '8.67', '3.47', '7.18', '8.45', '1.82', '1.94', '3.98']
first (string, 282 distinct): ['Robert', 'Christopher', 'Jessica', 'Michael', 'John', 'James', 'Samuel', 'Mary', 'Jennifer', 'Kenneth']
last (string, 388 distinct): ['Davis', 'Smith', 'Williams', 'Martinez', 'Johnson', 'Rodriguez', 'Fuller', 'Jones', 'Lowe', 'Bishop']
gender (string, 2 distinct): ['F', 'M']
street (string, 668 distinct): ['4293 Ramirez Squares', '72269 Elizabeth Field Apt. 132', '43235 Mckenzie Views Apt. 837', '444 Robert Mews', '428 Morgan River', '4130 Tiffany Glen Apt. 562', '594 Berry Lights Apt. 392', '329 Michael Extension', '8030 Beck Motorway', '37732 Joe Courts Apt. 752']
city (string, 626 distinct): ['Phoenix', 'Utica', 'San Antonio', 'Ranier', 'Fulton', 'Westport', 'Birmingham', 'Hudson', 'Clarks Mills', 'Thomas']
state (string, 48 distinct): ['NY', 'PA', 'TX', 'CA', 'OH', 'FL', 'MO', 'WV', 'MN', 'IL']
zip (numeric, 664 distinct): ['56668.0', '85020.0', '40077.0', '16114.0', '12534.0', '28405.0', '78248.0', '38761.0', '1843.0', '80120.0']
lat (numeric, 663 distinct): ['48.6031', '33.5623', '38.4921', '41.3851', '42.247', '29.5894', '34.2651', '33.4783', '39.5723', '39.5994']
long (numeric, 664 distinct): ['-93.2977', '-112.0559', '-85.4524', '-80.1752', '-73.7552', '-77.867', '-98.5201', '-90.5142', '-71.1605', '-105.0044']
city_pop (numeric, 619 distinct): ['1312922.0', '136.0', '1595797.0', '241.0', '564.0', '198.0', '1725.0', '1766.0', '1126.0', '606.0']
job (string, 349 distinct): ['Naval architect', 'Exhibition designer', 'Mechanical engineer', 'Film/video editor', 'Agricultural consultant', 'Ceramics designer', 'IT trainer', 'Petroleum engineer', 'Systems developer', 'Energy engineer']
dob (string, 658 distinct): ['1988-09-15', '2000-02-20', '1981-10-24', '1996-04-10', '1997-09-22', '1989-04-08', '1981-08-29', '1998-07-29', '1955-05-06', '1975-12-28']
unix_time (numeric, 5135 distinct): ['1386957227.0', '1386891618.0', '1386899318.0', '1387016966.0', '1387025413.0', '1386911932.0', '1386969507.0', '1386940912.0', '1387032681.0', '1386885655.0']
merch_lat (numeric, 5227 distinct): ['33.9864', '40.1852', '38.0462', '40.8552', '39.9311', '39.8929', '48.1835', '43.587', '45.9003', '43.0667']
merch_long (numeric, 5227 distinct): ['-81.2007', '-91.359', '-80.8002', '-78.9596', '-79.4938', '-87.0286', '-117.6799', '-76.0493', '-121.6104', '-89.6666']
'''

CONTEXT = "Transaction Fraud Detection for US Credit Card Transactions"
TARGET = CuratedTarget(raw_name="is_fraud", new_name="Fraudulent Transaction", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["trans_num", "unix_time"]
FEATURES = [CuratedFeature(raw_name="trans_date_trans_time", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="dob", new_name="Date Of Birth", feat_type=FeatureType.DATE)]
