from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wave_energy
====
Examples: 72000
====
URL: https://www.openml.org/search?type=data&id=44975
====
Description: **Data Description**

This data set consists of positions and absorbed power outputs of wave energy converters (WECs) in four real wave scenarios from the southern coast of Australia.

The data is obtained from an optimization method (blackbox optimization) with the goal of finding the optimal buoys placement.

Each instance represents wave energy returns for different placements of 16 buoys.

**Attribute Description**

1. *x[1-16]* - WECs positions (0-566)
2. *y[1-16]* - WECs positions (0-566)
3. *energy[1-16]* - WECs absorbed power (should be ignored if *energy_total* is considered as the target variable)
4. *energy_total* - total power output of the farm: Powerall, target feature
====
Target Variable: energy_total (numeric, 71993 distinct): ['3750339.773', '3813824.627', '3734448.005', '3721754.042', '3751779.569', '3644037.539', '3740987.273', '3648666.894', '3743372.696', '3786179.292']
====
Features:

x1 (numeric, 58248 distinct): ['566.0', '0.0', '565.6854', '211.0778', '261.4379', '199.8148', '299.3801', '150.9962', '339.4113', '251.1402']
x2 (numeric, 60168 distinct): ['0.0', '566.0', '565.6854', '380.0667', '74.1307', '119.5582', '396.7797', '164.3814', '435.2737', '123.2425']
x3 (numeric, 60223 distinct): ['0.0', '566.0', '565.6854', '401.1206', '505.4315', '56.5685', '194.717', '339.4113', '560.6399', '282.8427']
x4 (numeric, 59910 distinct): ['0.0', '566.0', '565.6854', '194.2219', '101.491', '17.8284', '322.9145', '509.1169', '453.8875', '50.4669']
x5 (numeric, 58153 distinct): ['566.0', '0.0', '565.6854', '46.9032', '507.4409', '214.6341', '430.2532', '244.0862', '46.5708', '217.6543']
x6 (numeric, 59301 distinct): ['0.0', '566.0', '565.6854', '395.1484', '508.6101', '344.0619', '56.5685', '403.1282', '320.8386', '1.4674']
x7 (numeric, 60758 distinct): ['0.0', '566.0', '565.6854', '71.7527', '321.0834', '389.3301', '2.7322', '339.4113', '509.1169', '282.8427']
x8 (numeric, 60915 distinct): ['0.0', '566.0', '565.6854', '347.7177', '163.5456', '152.4757', '420.7313', '56.5685', '87.8683', '241.8507']
x9 (numeric, 60305 distinct): ['0.0', '566.0', '565.6854', '308.0403', '239.9951', '521.8968', '206.1055', '509.1169', '360.9485', '124.5003']
x10 (numeric, 59705 distinct): ['0.0', '566.0', '565.6854', '513.7053', '219.849', '64.4156', '484.8966', '552.8919', '339.4113', '37.779']
x11 (numeric, 58960 distinct): ['0.0', '566.0', '565.6854', '494.0584', '245.8834', '121.5323', '209.2343', '233.498', '56.5685', '452.5483']
x12 (numeric, 57550 distinct): ['0.0', '566.0', '565.6854', '164.1692', '543.3506', '54.5677', '136.7449', '56.5685', '164.082', '113.1371']
x13 (numeric, 58693 distinct): ['0.0', '566.0', '565.6854', '427.9', '134.7489', '211.1858', '532.997', '270.9482', '5.3014', '56.5685']
x14 (numeric, 59247 distinct): ['0.0', '566.0', '565.6854', '484.256', '376.6677', '322.4725', '331.8632', '509.1169', '145.8304', '56.5685']
x15 (numeric, 60182 distinct): ['0.0', '566.0', '565.6854', '384.7296', '469.2127', '401.4417', '14.1407', '477.949', '195.0531', '113.1371']
x16 (numeric, 59851 distinct): ['566.0', '0.0', '565.6854', '511.4367', '416.2564', '445.5356', '370.0667', '56.5685', '399.4639', '563.094']
y1 (numeric, 58489 distinct): ['0.0', '566.0', '565.6854', '515.5035', '297.9808', '139.3985', '303.923', '551.4555', '501.8803', '339.4113']
y2 (numeric, 60861 distinct): ['0.0', '566.0', '565.6854', '447.2555', '163.8823', '106.2888', '172.0687', '509.1169', '265.7099', '433.8588']
y3 (numeric, 60607 distinct): ['0.0', '566.0', '565.6854', '507.3582', '116.1131', '501.9897', '25.1623', '267.0854', '509.1169', '56.5685']
y4 (numeric, 60573 distinct): ['566.0', '0.0', '565.6854', '192.9148', '426.7681', '462.7646', '19.8513', '274.2928', '283.8039', '179.7371']
y5 (numeric, 60032 distinct): ['0.0', '566.0', '565.6854', '257.6129', '76.6154', '401.4067', '8.7397', '154.9675', '527.8464', '339.4113']
y6 (numeric, 60328 distinct): ['0.0', '566.0', '565.6854', '51.5053', '399.5423', '516.1642', '528.0228', '537.9123', '420.2707', '147.4998']
y7 (numeric, 60907 distinct): ['0.0', '566.0', '565.6854', '164.5362', '375.9111', '56.5685', '330.7746', '113.1371', '93.5605', '446.748']
y8 (numeric, 60606 distinct): ['0.0', '566.0', '565.6854', '72.933', '563.1714', '56.5685', '40.0601', '113.1371', '131.5446', '169.7056']
y9 (numeric, 59689 distinct): ['0.0', '566.0', '565.6854', '63.1821', '223.5876', '238.7579', '56.5685', '157.257', '173.8018', '226.2742']
y10 (numeric, 59838 distinct): ['0.0', '566.0', '565.6854', '4.0922', '232.11', '540.4266', '490.0804', '352.7587', '509.1169', '479.1709']
y11 (numeric, 59162 distinct): ['0.0', '566.0', '565.6854', '280.4238', '550.4054', '438.9473', '441.1297', '16.487', '56.5685', '509.1169']
y12 (numeric, 59248 distinct): ['0.0', '566.0', '565.6854', '462.7117', '191.4111', '445.402', '435.8791', '56.5685', '258.0115', '230.004']
y13 (numeric, 59798 distinct): ['0.0', '566.0', '565.6854', '96.7775', '196.4874', '381.5395', '56.5685', '54.6682', '509.1169', '226.2742']
y14 (numeric, 60547 distinct): ['0.0', '566.0', '565.6854', '471.3878', '509.1169', '110.9388', '146.905', '339.4113', '172.4103', '434.2491']
y15 (numeric, 61057 distinct): ['566.0', '0.0', '565.6854', '302.6084', '222.6024', '18.5582', '565.1745', '56.5685', '32.2273', '509.1169']
y16 (numeric, 61036 distinct): ['0.0', '566.0', '565.6854', '74.1922', '532.509', '93.7857', '544.0166', '173.389', '509.1169', '29.3683']
energy1 (numeric, 71985 distinct): ['236386.7206', '256731.9481', '167386.0988', '200669.2793', '257135.1206', '270260.7042', '246225.4083', '211826.463', '207506.2846', '185730.0531']
energy2 (numeric, 71981 distinct): ['274458.3372', '253652.3775', '223922.0829', '279073.8555', '240898.9837', '268671.1436', '210483.1471', '268372.1633', '210793.1167', '266089.8707']
energy3 (numeric, 71986 distinct): ['264468.0601', '251012.4563', '259129.0001', '280485.4509', '269109.2217', '261241.811', '227752.1417', '263183.6383', '198668.6008', '208858.6777']
energy4 (numeric, 71982 distinct): ['235915.5033', '244634.3314', '237162.2628', '270563.9884', '200080.4368', '264025.452', '265236.9004', '235486.0075', '266726.2064', '265823.8636']
energy5 (numeric, 71988 distinct): ['281754.3107', '274430.2049', '273151.7856', '234744.6468', '273356.8127', '266064.7978', '272070.0874', '276633.9015', '231440.1316', '269196.6198']
energy6 (numeric, 71986 distinct): ['265784.7546', '272677.8828', '271553.1287', '265225.5266', '221029.3249', '255168.0523', '228353.4844', '271861.4143', '268108.6773', '276268.9392']
energy7 (numeric, 71983 distinct): ['271321.6326', '269905.3626', '276538.2536', '279812.8298', '270811.5619', '216856.3012', '276785.1805', '239600.346', '278336.4291', '269557.7101']
energy8 (numeric, 71986 distinct): ['276762.6303', '252870.6488', '257126.4956', '206790.9825', '219719.8631', '189103.3097', '274015.8416', '271013.0087', '233042.8776', '266619.851']
energy9 (numeric, 71980 distinct): ['268049.6926', '277237.63', '270642.3568', '167359.3977', '271330.7541', '216289.836', '273073.2235', '267399.882', '265676.2688', '276632.7584']
energy10 (numeric, 71983 distinct): ['255821.4456', '248916.4717', '200414.2244', '236229.5865', '275282.5051', '236731.1034', '213717.5207', '273026.4445', '258111.4349', '265362.3194']
energy11 (numeric, 71987 distinct): ['271099.3219', '265465.8798', '274795.565', '247097.5306', '266325.1224', '269929.5036', '271077.596', '220454.9068', '208927.8629', '271888.5102']
energy12 (numeric, 71987 distinct): ['260704.928', '275019.7401', '263565.2985', '267784.0117', '267100.712', '248817.8696', '266316.8833', '239428.9364', '221439.9508', '231199.8442']
energy13 (numeric, 71982 distinct): ['265009.2632', '192193.3586', '255337.7659', '272676.822', '255221.9815', '256594.8286', '265616.3056', '219171.9539', '270998.847', '264776.3698']
energy14 (numeric, 71985 distinct): ['249432.8993', '218393.8535', '251521.7909', '264957.9945', '272117.7627', '196288.9596', '229535.8112', '259739.181', '260259.0771', '271598.2568']
energy15 (numeric, 71984 distinct): ['217827.831', '188349.6398', '277741.2538', '223420.3353', '236766.815', '224779.0276', '247083.3803', '247490.5374', '272883.0268', '201518.4757']
energy16 (numeric, 71983 distinct): ['265906.2956', '228202.6434', '265476.7086', '265230.0649', '266712.5726', '189160.257', '258493.4339', '267528.1326', '265218.6925', '186997.8425']
'''

CONTEXT = "Wave Energy Converters in Southern Coast of Australia"
TARGET = CuratedTarget(raw_name="energy_total", new_name="Total Power Output of the Farm",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["energy1", "energy2", "energy3", "energy4", "energy5", "energy6", "energy7", "energy8", "energy9",
                "energy10", "energy11", "energy12", "energy13", "energy14", "energy15", "energy16"]
FEATURES = [CuratedFeature(raw_name="x1", new_name="X - WEC 1 Position"),
            CuratedFeature(raw_name="x2", new_name="X - WEC 2 Position"),
            CuratedFeature(raw_name="x3", new_name="X - WEC 3 Position"),
            CuratedFeature(raw_name="x4", new_name="X - WEC 4 Position"),
            CuratedFeature(raw_name="x5", new_name="X - WEC 5 Position"),
            CuratedFeature(raw_name="x6", new_name="X - WEC 6 Position"),
            CuratedFeature(raw_name="x7", new_name="X - WEC 7 Position"),
            CuratedFeature(raw_name="x8", new_name="X - WEC 8 Position"),
            CuratedFeature(raw_name="x9", new_name="X - WEC 9 Position"),
            CuratedFeature(raw_name="x10", new_name="X - WEC 10 Position"),
            CuratedFeature(raw_name="x11", new_name="X - WEC 11 Position"),
            CuratedFeature(raw_name="x12", new_name="X - WEC 12 Position"),
            CuratedFeature(raw_name="x13", new_name="X - WEC 13 Position"),
            CuratedFeature(raw_name="x14", new_name="X - WEC 14 Position"),
            CuratedFeature(raw_name="x15", new_name="X - WEC 15 Position"),
            CuratedFeature(raw_name="x16", new_name="X - WEC 16 Position"),
            CuratedFeature(raw_name="y1", new_name="Y - WEC 1 Position"),
            CuratedFeature(raw_name="y2", new_name="Y - WEC 2 Position"),
            CuratedFeature(raw_name="y3", new_name="Y - WEC 3 Position"),
            CuratedFeature(raw_name="y4", new_name="Y - WEC 4 Position"),
            CuratedFeature(raw_name="y5", new_name="Y - WEC 5 Position"),
            CuratedFeature(raw_name="y6", new_name="Y - WEC 6 Position"),
            CuratedFeature(raw_name="y7", new_name="Y - WEC 7 Position"),
            CuratedFeature(raw_name="y8", new_name="Y - WEC 8 Position"),
            CuratedFeature(raw_name="y9", new_name="Y - WEC 9 Position"),
            CuratedFeature(raw_name="y10", new_name="Y - WEC 10 Position"),
            CuratedFeature(raw_name="y11", new_name="Y - WEC 11 Position"),
            CuratedFeature(raw_name="y12", new_name="Y - WEC 12 Position"),
            CuratedFeature(raw_name="y13", new_name="Y - WEC 13 Position"),
            CuratedFeature(raw_name="y14", new_name="Y - WEC 14 Position"),
            CuratedFeature(raw_name="y15", new_name="Y - WEC 15 Position"),
            CuratedFeature(raw_name="y16", new_name="Y - WEC 16 Position"),
            ]