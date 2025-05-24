from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: sulfur
====
Examples: 10081
====
URL: https://www.openml.org/search?type=data&id=44145
====
Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark, transformed in the same way. This dataset belongs to the "regression on numerical features" benchmark. Original description: 
 
&quot;The sulfur recovery unit (SRU) removes environmental pollutants from acid gas
streams before they are released into the atmosphere. Furthermore, elemental sulfur
is recovered as a valuable by-product.&quot;

5 inputs variables are gas and air flows.
2 outputs to predict are H2S and SO2 concentrations

See Appendix A.5 of Fortuna, L., Graziani, S., Rizzo, A., Xibilia, M.G. &quot;Soft Sensors for Monitoring and Control of Industrial Processes&quot; (Springer 2007) for more info.
====
Target Variable: y1 (numeric, 9368 distinct): ['0.078', '0.0861', '0.072', '0.0475', '0.0749', '0.0493', '0.1062', '0.0556', '0.0896', '0.0257']
====
Features:

a1 (numeric, 9568 distinct): ['0.489', '0.4036', '0.8255', '0.2658', '0.6627', '0.8385', '0.4162', '0.8168', '0.5136', '0.8578']
a2 (numeric, 8249 distinct): ['0.5046', '0.5149', '0.8961', '0.597', '0.3364', '0.8967', '0.898', '0.8976', '0.8995', '0.5102']
a3 (numeric, 9839 distinct): ['0.5454', '0.4554', '0.4548', '0.3503', '0.488', '0.464', '0.4838', '0.4316', '0.4825', '0.4527']
a4 (numeric, 7561 distinct): ['0.7342', '0.7306', '0.7217', '0.7258', '0.7295', '0.7219', '0.733', '0.7238', '0.7346', '0.7539']
a5 (numeric, 6923 distinct): ['0.703', '0.7013', '0.6991', '0.7002', '0.697', '0.7003', '0.702', '0.6964', '0.6974', '0.6996']
y2 (numeric, 9678 distinct): ['0.1767', '0.0025', '0.0166', '0.0101', '0.1571', '0.0159', '0.1529', '0.1647', '0.1735', '0.1854']
'''

CONTEXT = "Sulfur Recovery Unit (SRU) Process prediction by gas and air flows"
TARGET = CuratedTarget(raw_name="y1", new_name="H2S Concentration", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['y2']
FEATURES = []