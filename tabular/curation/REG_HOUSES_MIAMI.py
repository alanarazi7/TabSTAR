from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: miami_housing
====
Examples: 13932
====
URL: https://www.openml.org/search?type=data&id=44983
====
Description: **Data description**

The dataset contains information on 13,932 single-family homes sold in Miami .
Content.

The goal is to predict the sale price.

**Attribute description**

The dataset contains the following columns:

  * PARCELNO: unique identifier for each property. About 1% appear multiple times.
  * SALE_PRC: sale price ($)
  * LND_SQFOOT: land area (square feet)
  * TOTLVGAREA: floor area (square feet)
  * SPECFEATVAL: value of special features (e.g., swimming pools) ($)
  * RAIL_DIST: distance to the nearest rail line (an indicator of noise) (feet)
  * OCEAN_DIST: distance to the ocean (feet)
  * WATER_DIST: distance to the nearest body of water (feet)
  * CNTR_DIST: distance to the Miami central business district (feet)
  * SUBCNTR_DI: distance to the nearest subcenter (feet)
  * HWY_DIST: distance to the nearest highway (an indicator of noise) (feet)
  * age: age of the structure
  * avno60plus: dummy variable for airplane noise exceeding an acceptable level
  * structure_quality: quality of the structure
  * month_sold: sale month in 2016 (1 = jan)
  * LATITUDE
  * LONGITUDE
====
Target Variable: SALE_PRC (numeric, 2111 distinct): ['250000.0', '300000.0', '260000.0', '270000.0', '280000.0', '290000.0', '350000.0', '265000.0', '285000.0', '210000.0']
====
Features:

LATITUDE (numeric, 13776 distinct): ['25.8375', '25.8166', '25.7496', '25.9081', '25.7194', '25.8617', '25.713', '25.8549', '25.8861', '25.9465']
LONGITUDE (numeric, 13776 distinct): ['-80.2289', '-80.1984', '-80.4315', '-80.168', '-80.4426', '-80.1967', '-80.4449', '-80.1772', '-80.2257', '-80.1607']
LND_SQFOOT (numeric, 4696 distinct): ['7500.0', '5000.0', '6000.0', '7875.0', '8250.0', '8000.0', '15000.0', '5500.0', '5250.0', '10000.0']
TOT_LVG_AREA (numeric, 2978 distinct): ['3079.0', '3199.0', '1440.0', '2176.0', '1701.0', '2578.0', '2193.0', '2091.0', '1606.0', '2514.0']
SPEC_FEAT_VAL (numeric, 7583 distinct): ['0.0', '550.0', '440.0', '4800.0', '1200.0', '3200.0', '2240.0', '2200.0', '2460.0', '1296.0']
RAIL_DIST (numeric, 13235 distinct): ['50.0', '16135.4', '49.9', '7140.5', '2510.4', '699.9', '6558.1', '2549.3', '1802.3', '14690.8']
OCEAN_DIST (numeric, 13617 distinct): ['21012.0', '42047.0', '55025.2', '34891.0', '13858.2', '22211.5', '61433.2', '15183.3', '28968.2', '33545.3']
WATER_DIST (numeric, 13218 distinct): ['0.0', '7.2', '3424.0', '11.0', '6047.0', '6252.7', '1831.1', '2292.4', '31855.7', '523.4']
CNTR_DIST (numeric, 13682 distinct): ['43025.9', '79187.6', '14557.6', '25074.0', '93953.7', '48378.1', '88949.5', '88348.4', '60642.1', '35167.4']
SUBCNTR_DI (numeric, 13642 distinct): ['60514.7', '45214.2', '37728.9', '44742.7', '25074.0', '14557.6', '44362.6', '45275.0', '29573.8', '47732.3']
HWY_DIST (numeric, 13213 distinct): ['2140.8', '1022.6', '857.4', '13816.9', '4589.9', '13752.4', '1848.0', '10656.3', '4522.1', '1582.4']
age (numeric, 96 distinct): ['0', '16', '26', '21', '11', '36', '12', '10', '23', '31']
avno60plus (numeric, 2 distinct): ['0', '1']
month_sold (numeric, 12 distinct): ['6', '8', '5', '4', '3', '9', '7', '12', '11', '10']
structure_quality (numeric, 5 distinct): ['4', '2', '5', '1', '3']
'''

CONTEXT = "Family Houses sold in Miami"
TARGET = CuratedTarget(raw_name="SALE_PRC", new_name="Sale Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []