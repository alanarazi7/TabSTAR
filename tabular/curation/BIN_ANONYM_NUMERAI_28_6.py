from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: numerai28.6
====
Examples: 96320
====
URL: https://www.openml.org/search?type=data&id=23517
====
Description: **Author**: Numer.ai  
**Source**: [Kaggle](https://www.kaggle.com/numerai/encrypted-stock-market-data-from-numerai)  
**Please cite**:   

**Encrypted Stock Market Training Data from Numer.ai**  
The data is cleaned, regularized and encrypted global equity data. The first 21 columns (feature1 - feature21) are features, and target is the binary class you’re trying to predict.
====
Target Variable: attribute_21 (nominal, 2 distinct): ['1', '0']
====
Features:

attribute_0 (numeric, 1001 distinct): ['0.7256', '0.7564', '0.5931', '0.4356', '0.7997', '0.4935', '0.4101', '0.339', '0.8283', '0.4629']
attribute_1 (numeric, 1001 distinct): ['0.1207', '0.9324', '0.4262', '0.9169', '0.4658', '0.0181', '0.1375', '0.054', '0.7781', '0.4575']
attribute_2 (numeric, 1000 distinct): ['0.317', '0.233', '0.8996', '0.6997', '0.5576', '0.8804', '0.0405', '0.0079', '0.8012', '0.6307']
attribute_3 (numeric, 1001 distinct): ['0.9694', '0.7097', '0.4839', '0.0282', '0.892', '0.2469', '0.0995', '0.0006', '0.9598', '0.9922']
attribute_4 (numeric, 1000 distinct): ['0.1852', '0.0753', '0.8862', '0.0029', '0.3325', '0.9361', '0.4333', '0.5959', '0.0602', '0.2025']
attribute_5 (numeric, 1001 distinct): ['0.3974', '0.3066', '0.7854', '0.1363', '0.0792', '0.6693', '0.9855', '0.1197', '0.3418', '0.6156']
attribute_6 (numeric, 1000 distinct): ['0.0687', '0.1607', '0.9847', '0.4206', '0.96', '0.2692', '0.8383', '0.1857', '0.8744', '0.3864']
attribute_7 (numeric, 1001 distinct): ['0.0386', '0.7382', '0.9033', '0.0319', '0.3585', '0.6123', '0.4693', '0.1951', '0.4806', '0.7514']
attribute_8 (numeric, 1001 distinct): ['0.7004', '0.7702', '0.2497', '0.9746', '0.8085', '0.1979', '0.5587', '0.3593', '0.0002', '0.9447']
attribute_9 (numeric, 1000 distinct): ['0.0488', '0.9431', '0.9111', '0.0387', '0.5331', '0.0398', '0.537', '0.7532', '0.0197', '0.0173']
attribute_10 (numeric, 1001 distinct): ['0.0245', '0.0209', '0.7609', '0.0067', '0.0402', '0.5118', '0.0287', '0.0341', '0.579', '0.0058']
attribute_11 (numeric, 1001 distinct): ['0.0125', '0.0795', '0.1822', '0.907', '0.0071', '0.3175', '0.909', '0.0127', '0.0699', '0.4182']
attribute_12 (numeric, 1000 distinct): ['0.9316', '0.1602', '0.9219', '0.0738', '0.9102', '0.8948', '0.1839', '0.0739', '0.9219', '0.4353']
attribute_13 (numeric, 1001 distinct): ['0.9729', '0.5173', '0.9547', '0.2523', '0.7014', '0.1422', '0.3744', '0.7017', '0.9757', '0.2514']
attribute_14 (numeric, 1001 distinct): ['0.0484', '0.5589', '0.8486', '0.9362', '0.3254', '0.4969', '0.4738', '0.8471', '0.9179', '0.3355']
attribute_15 (numeric, 1000 distinct): ['0.9902', '0.7288', '0.6935', '0.0302', '0.4603', '0.8588', '0.8194', '0.5546', '0.0177', '0.7291']
attribute_16 (numeric, 1001 distinct): ['0.2418', '0.6475', '0.4396', '0.9883', '0.2375', '0.7442', '0.1503', '0.2555', '0.1311', '0.7912']
attribute_17 (numeric, 1001 distinct): ['0.2116', '0.7801', '0.5482', '0.8851', '0.0054', '0.017', '0.3398', '0.0129', '0.1774', '0.6157']
attribute_18 (numeric, 1001 distinct): ['0.8738', '0.01', '0.9005', '0.8985', '0.857', '0.0152', '0.2129', '0.1343', '0.0479', '0.453']
attribute_19 (numeric, 1001 distinct): ['0.4869', '0.0342', '0.8007', '0.9698', '0.4975', '0.6167', '0.7083', '0.0127', '0.1992', '0.4771']
attribute_20 (numeric, 1001 distinct): ['0.1314', '0.3378', '0.9569', '0.0013', '0.0636', '0.9734', '0.5081', '0.603', '0.6204', '0.573']
'''

CONTEXT = "Encrypted Stock Market Training Data from Numer.ai"
TARGET = CuratedTarget(raw_name="attribute_21", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []