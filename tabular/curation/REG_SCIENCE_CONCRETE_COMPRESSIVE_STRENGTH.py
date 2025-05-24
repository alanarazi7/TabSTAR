from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: concrete_compressive_strength
====
Examples: 1030
====
URL: https://www.openml.org/search?type=data&id=44959
====
Description: **Data Description**

Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. Each instance represents a description, different features of concrete instance, including its compressive strength. The latter can be predicted using the other features about the concrete.

**Attribute Description**

1. *cement* - amount in kg in a m3 mixture
2. *blast_furnace_slag* - amount in kg in a m3 mixture
3. *fly_ash* - amount in kg in a m3 mixture
4. *water* - amount in kg in a m3 mixture
5. *superplasticizer* - amount in kg in a m3 mixture
6. *coarse_aggregate* - amount in kg in a m3 mixture
7. *fine_aggregate* - amount in kg in a m3 mixture
8. *age* - age in days (1 - 365)
9. *strength* - in MPa, target feature
====
Target Variable: strength (numeric, 938 distinct): ['33.3982', '77.2972', '31.3505', '71.2987', '35.3012', '79.2966', '55.8958', '17.5403', '18.1263', '65.1969']
====
Features:

cement (numeric, 280 distinct): ['425.0', '362.6', '251.37', '446.0', '310.0', '331.0', '250.0', '475.0', '387.0', '349.0']
blast_furnace_slag (numeric, 187 distinct): ['0.0', '189.0', '106.3', '24.0', '20.0', '145.0', '19.0', '22.0', '26.0', '190.0']
fly_ash (numeric, 163 distinct): ['0.0', '141.0', '118.27', '79.0', '94.0', '174.24', '98.75', '95.69', '125.18', '121.62']
water (numeric, 205 distinct): ['192.0', '228.0', '185.7', '203.5', '186.0', '162.0', '164.9', '185.0', '153.5', '200.0']
superplasticizer (numeric, 155 distinct): ['0.0', '8.0', '11.6', '7.0', '6.0', '9.0', '16.5', '10.0', '11.0', '5.75']
coarse_aggregate (numeric, 284 distinct): ['932.0', '852.1', '944.7', '968.0', '1125.0', '1047.0', '967.0', '974.0', '942.0', '938.0']
fine_aggregate (numeric, 304 distinct): ['755.8', '594.0', '670.0', '613.0', '801.0', '746.6', '887.1', '845.0', '712.0', '750.0']
age (numeric, 14 distinct): ['28.0', '3.0', '7.0', '56.0', '14.0', '90.0', '100.0', '180.0', '91.0', '365.0']
'''

CONTEXT = "Concrete Compressive Strength"
TARGET = CuratedTarget(raw_name="strength", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []