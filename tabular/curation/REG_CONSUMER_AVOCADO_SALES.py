from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: avocado-sales
====
Examples: 18249
====
URL: https://www.openml.org/search?type=data&id=41210
====
Description: Historical data on avocado prices and sales volume in multiple US markets. Downloaded from Kaggle [https://www.kaggle.com/neuromusic/avocado-prices/home] on 29.10.2018. The original data stems from the HASS AVOCADO BOARD [http://www.hassavocadoboard.com/retail/volume-and-price-data]. The Kaggle dataset was licensed under the Open Database License (ODbL) [https://opendatacommons.org/licenses/odbl/1.0/]. The variable 'AveragePrice' was selected as target variable. For a description of all variables checkout the Kaggle dataset repo or the original dataset description by the HASS AVOCADO BOARD. 'Year' is coded as a categorical features as the dataset covers only the years 2015-2018. The dataset also includes a 'Date' variable (ignored by default) which can be used to construct additional month or day features. The ID variable from the Kaggle version was removed from the dataset.
====
Target Variable: AveragePrice (numeric, 259 distinct): ['1.15', '1.18', '1.08', '1.26', '1.13', '0.98', '1.19', '1.36', '1.59', '1.43']
====
Features:

Total Volume (numeric, 18237 distinct): ['4103.97', '3529.44', '46602.16', '13234.04', '3713.49', '19634.24', '3288.85', '9465.99', '2038.99', '2858.31']
4046 (numeric, 17702 distinct): ['0.0', '3.0', '4.0', '1.24', '1.0', '1.25', '6.0', '1.21', '1.3', '1.27']
4225 (numeric, 18103 distinct): ['0.0', '177.87', '215.36', '1.3', '1.26', '94.74', '13.6', '20.32', '35898.69', '6973.51']
4770 (numeric, 12071 distinct): ['0.0', '2.66', '3.32', '10.97', '1.59', '1.64', '1.6', '2.74', '1.66', '1.18']
Total Bags (numeric, 18097 distinct): ['0.0', '990.0', '300.0', '550.0', '266.67', '916.67', '286.67', '263.33', '196.67', '260.0']
Small Bags (numeric, 17321 distinct): ['0.0', '203.33', '223.33', '533.33', '123.33', '196.67', '70.0', '103.33', '216.67', '20.0']
Large Bags (numeric, 15082 distinct): ['0.0', '3.33', '6.67', '10.0', '4.44', '13.33', '16.67', '26.67', '6.66', '20.0']
XLarge Bags (numeric, 5588 distinct): ['0.0', '3.33', '6.67', '1.11', '5.0', '10.0', '16.67', '2.22', '150.0', '13.33']
type (nominal, 2 distinct): ['conventional', 'organic']
year (nominal, 4 distinct): ['2017', '2016', '2015', '2018']
region (nominal, 54 distinct): ['Albany', 'Sacramento', 'Northeast', 'NorthernNewEngland', 'Orlando', 'Philadelphia', 'PhoenixTucson', 'Pittsburgh', 'Plains', 'Portland']
'''

CONTEXT = "Avocado Sales in multiple US markets"
TARGET = CuratedTarget(raw_name="AveragePrice", new_name="Average Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []