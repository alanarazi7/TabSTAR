from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Higgs
====
Examples: 1000000
====
URL: https://www.openml.org/search?type=data&id=42769
====
Description: This is a smaller version of the original dataset, containing 1M rows. 
**Author**: Daniel Whiteson, University of California Irvine  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/HIGGS)  
**Please cite**: Baldi, P., P. Sadowski, and D. Whiteson. Searching for Exotic Particles in High-energy Physics with Deep Learning. Nature Communications 5 (July 2, 2014).  

**Higgs Boson detection data**. The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. The last 500,000 examples are used as a test set.

**Note: This is the UCI Higgs dataset, same as version 1, but it fixes the definition of the class attribute, which is categorical, not numeric.**


### Attribute Information
* The first column is the class label (1 for signal, 0 for background)
* 21 low-level features (kinematic properties): lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag
* 7 high-level features derived by physicists: m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb. 

For more detailed information about each feature see the original paper.

Relevant Papers:

Baldi, P., P. Sadowski, and D. Whiteson. Searching for Exotic Particles in High-energy Physics with Deep Learning. Nature Communications 5 (July 2, 2014).
====
Target Variable: target (nominal, 2 distinct): ['1', '0']
====
Features:

lepton_pT (numeric, 19815 distinct): ['0.6182', '0.7317', '0.5651', '0.6632', '0.5864', '0.6429', '0.7295', '0.6365', '0.7064', '0.6043']
lepton_eta (numeric, 5001 distinct): ['0.1071', '-0.188', '-0.3857', '-0.0449', '-0.2065', '0.3136', '-0.2552', '0.1217', '-0.1491', '-0.1822']
lepton_phi (numeric, 6284 distinct): ['0.8138', '0.6246', '-0.0801', '0.5902', '-1.5067', '-0.8037', '-1.449', '-1.4224', '1.3998', '0.6296']
missing_energy_magnitude (numeric, 616259 distinct): ['1.7487', '1.7308', '1.7191', '1.7658', '1.8839', '1.7512', '1.8413', '1.7216', '1.782', '1.7493']
missing_energy_phi (numeric, 637068 distinct): ['-1.5436', '1.1762', '0.6993', '0.7445', '-1.3277', '0.7211', '-0.8709', '-1.377', '1.4469', '0.7775']
jet_1_pt (numeric, 34131 distinct): ['0.762', '0.7729', '0.8868', '0.7741', '0.7799', '0.7282', '0.7588', '0.8112', '0.8203', '0.7264']
jet_1_eta (numeric, 5999 distinct): ['0.1485', '-0.1486', '0.0515', '0.0495', '0.0455', '0.1495', '0.0485', '-0.4456', '-0.0485', '-0.4476']
jet_1_phi (numeric, 6284 distinct): ['-1.3543', '-0.6907', '1.1887', '0.9127', '-0.8565', '-0.4146', '0.3594', '-0.0277', '-1.4097', '0.3045']
jet_1_b-tag (numeric, 3 distinct): ['0.0', '2.1731', '1.0865']
jet_2_pt (numeric, 26960 distinct): ['0.7257', '0.6854', '0.692', '0.6808', '0.6988', '0.658', '0.712', '0.8146', '0.8428', '0.7797']
jet_2_eta (numeric, 5999 distinct): ['-0.2428', '-0.0485', '-0.1456', '0.0486', '0.0583', '0.2429', '-0.234', '0.1458', '-0.1447', '0.139']
jet_2_phi (numeric, 6284 distinct): ['0.4155', '1.301', '-0.5251', '0.36', '-0.0823', '0.2496', '-1.4661', '0.471', '0.3051', '-0.9678']
jet_2_b-tag (numeric, 3 distinct): ['0.0', '2.2149', '1.1074']
jet_3_pt (numeric, 18733 distinct): ['0.7747', '0.71', '0.8163', '0.6626', '0.6981', '0.7757', '0.8226', '0.7004', '0.7559', '0.7174']
jet_3_eta (numeric, 5999 distinct): ['0.0457', '-0.1364', '-0.0453', '0.1367', '0.2277', '0.4098', '0.3188', '-0.0426', '0.0557', '0.1495']
jet_3_phi (numeric, 6284 distinct): ['1.1898', '-0.9677', '-1.2445', '0.3599', '-0.9122', '-0.3036', '1.1349', '-1.4103', '-1.7398', '1.08']
jet_3_b-tag (numeric, 3 distinct): ['0.0', '2.5482', '1.2741']
jet_4_pt (numeric, 14035 distinct): ['0.6136', '0.6032', '0.5732', '0.6312', '0.6662', '0.6268', '0.5657', '0.5603', '0.4941', '0.6343']
jet_4_eta (numeric, 5999 distinct): ['0.037', '0.1253', '-0.2137', '-0.1221', '-0.0354', '-0.3602', '0.2602', '-0.1779', '0.1978', '0.5192']
jet_4_phi (numeric, 6284 distinct): ['1.6329', '0.4493', '-1.4109', '0.1941', '-1.5213', '0.7817', '0.36', '0.9149', '-0.1024', '-0.5702']
jet_4_b-tag (numeric, 3 distinct): ['0.0', '3.102', '1.551']
m_jj (numeric, 502300 distinct): ['0.9237', '1.1515', '0.8328', '1.1504', '1.2485', '0.8848', '1.1417', '1.1959', '1.1462', '0.8569']
m_jjj (numeric, 213126 distinct): ['0.9369', '0.9297', '0.9609', '0.9675', '0.9073', '0.9231', '0.9724', '0.8933', '0.929', '0.9312']
m_lv (numeric, 185039 distinct): ['0.9878', '0.9875', '0.9872', '0.9874', '0.9875', '0.9877', '0.9874', '0.9875', '0.9876', '0.9873']
m_jlv (numeric, 260395 distinct): ['0.8579', '0.8847', '0.9258', '0.8715', '0.9028', '0.8738', '0.859', '0.8608', '0.7881', '0.8309']
m_bb (numeric, 455299 distinct): ['0.8749', '0.8945', '0.8538', '0.9153', '0.8771', '0.8512', '0.8241', '0.9217', '0.8969', '0.9468']
m_wbb (numeric, 352577 distinct): ['0.9046', '0.9442', '0.891', '0.9055', '0.9712', '0.8898', '0.9606', '0.8742', '0.9364', '0.917']
m_wwbb (numeric, 406261 distinct): ['0.875', '0.7978', '0.8452', '0.8065', '0.8076', '0.8134', '0.7937', '0.7835', '0.7967', '0.8369']
'''

CONTEXT = "Particle Higgs Boson Detection"
TARGET = CuratedTarget(raw_name="target", new_name="Signal", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': "Background", '1': "Signal"})
COLS_TO_DROP = []
FEATURES = []