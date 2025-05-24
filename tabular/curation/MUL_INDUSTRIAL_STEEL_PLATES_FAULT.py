from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: steel-plates-fault
====
Examples: 1941
====
URL: https://www.openml.org/search?type=data&id=40982
====
Description: **Author**: Semeion, Research Center of Sciences of Communication, Rome, Italy.     
**Source**: [UCI](http://archive.ics.uci.edu/ml/datasets/steel+plates+faults)     
**Please cite**: Dataset provided by Semeion, Research Center of Sciences of Communication, Via Sersale 117, 00128, Rome, Italy.  

__Changes w.r.t. version 1: included one target factor with 7 levels as target variable for the classification. Also deleted the previous 7 binary target variables.__

**Steel Plates Faults Data Set**  
A dataset of steel plates' faults, classified into 7 different types. The goal was to train machine learning for automatic pattern recognition.

The dataset consists of 27 features describing each fault (location, size, ...) and 1 feature indicating the type of fault (on of 7: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults). The target is the type of fault.

### Attribute Information  
* V1: X_Minimum  
* V2: X_Maximum  
* V3: Y_Minimum  
* V4: Y_Maximum  
* V5: Pixels_Areas  
* V6: X_Perimeter  
* V7: Y_Perimeter  
* V8: Sum_of_Luminosity  
* V9: Minimum_of_Luminosity  
* V10: Maximum_of_Luminosity  
* V11: Length_of_Conveyer  
* V12: TypeOfSteel_A300  
* V13: TypeOfSteel_A400  
* V14: Steel_Plate_Thickness  
* V15: Edges_Index  
* V16: Empty_Index  
* V17: Square_Index  
* V18: Outside_X_Index  
* V19: Edges_X_Index  
* V20: Edges_Y_Index  
* V21: Outside_Global_Index  
* V22: LogOfAreas  
* V23: Log_X_Index  
* V24: Log_Y_Index  
* V25: Orientation_Index  
* V26: Luminosity_Index  
* V27: SigmoidOfAreas  
* target: 7 types of fault as classification target  

### Relevant Papers  
1.M Buscema, S Terzi, W Tastle, A New Meta-Classifier,in NAFIPS 2010, Toronto (CANADA),26-28 July 2010, 978-1-4244-7858-6/10 Â©2010 IEEE  
2.M Buscema, MetaNet: The Theory of Independent Judges, in Substance Use & Misuse, 33(2), 439-461,1998
====
Target Variable: target (nominal, 7 distinct): ['Other_Faults', 'Bumps', 'K_Scratch', 'Z_Scratch', 'Pastry', 'Stains', 'Dirtiness']
====
Features:

V1 (numeric, 962 distinct): ['41.0', '39.0', '0.0', '43.0', '37.0', '2.0', '19.0', '9.0', '13.0', '15.0']
V2 (numeric, 994 distinct): ['212.0', '214.0', '218.0', '216.0', '194.0', '211.0', '192.0', '209.0', '193.0', '210.0']
V3 (numeric, 1939 distinct): ['1803992.0', '28972.0', '270900.0', '430948.0', '409986.0', '402394.0', '393457.0', '379740.0', '375592.0', '363701.0']
V4 (numeric, 1940 distinct): ['28984.0', '270944.0', '1229628.0', '410023.0', '402418.0', '393488.0', '379759.0', '375611.0', '363718.0', '149070.0']
V5 (numeric, 920 distinct): ['52.0', '68.0', '60.0', '55.0', '51.0', '16.0', '74.0', '110.0', '63.0', '67.0']
V6 (numeric, 399 distinct): ['12.0', '15.0', '13.0', '14.0', '11.0', '16.0', '10.0', '9.0', '17.0', '18.0']
V7 (numeric, 317 distinct): ['11.0', '12.0', '10.0', '13.0', '14.0', '17.0', '15.0', '20.0', '8.0', '16.0']
V8 (numeric, 1909 distinct): ['8140.0', '41476.0', '13352.0', '16308.0', '6216.0', '10024.0', '29002.0', '13351.0', '11890.0', '7502.0']
V9 (numeric, 161 distinct): ['101', '91', '97', '104', '96', '95', '99', '84', '105', '120']
V10 (numeric, 100 distinct): ['127', '126', '124', '141', '125', '132', '143', '140', '134', '135']
V11 (numeric, 84 distinct): ['1358.0', '1356.0', '1360.0', '1362.0', '1364.0', '1692.0', '1353.0', '1687.0', '1354.0', '1387.0']
V12 (numeric, 2 distinct): ['0', '1']
V13 (numeric, 2 distinct): ['1', '0']
V14 (numeric, 24 distinct): ['40.0', '70.0', '100.0', '80.0', '50.0', '60.0', '200.0', '300.0', '69.0', '175.0']
V15 (numeric, 1387 distinct): ['0.0604', '0.0', '0.0574', '0.0557', '0.0585', '0.0605', '0.0556', '0.0586', '0.0575', '0.0558']
V16 (numeric, 1338 distinct): ['0.3333', '0.25', '0.2222', '0.3636', '0.2', '0.375', '0.4', '0.5', '0.3', '0.2778']
V17 (numeric, 770 distinct): ['1.0', '0.8', '0.5', '0.6667', '0.3333', '0.75', '0.8889', '0.9091', '0.8571', '0.4']
V18 (numeric, 454 distinct): ['0.0059', '0.0066', '0.0081', '0.0088', '0.0053', '0.0074', '0.0044', '0.0047', '0.0065', '0.0052']
V19 (numeric, 818 distinct): ['1.0', '0.8', '0.75', '0.6667', '0.5', '0.8333', '0.9', '0.8571', '0.7778', '0.8889']
V20 (numeric, 648 distinct): ['1.0', '0.6667', '0.9231', '0.8333', '0.75', '0.9091', '0.8', '0.9167', '0.9', '0.875']
V21 (numeric, 3 distinct): ['1.0', '0.0', '0.5']
V22 (numeric, 914 distinct): ['1.8325', '1.716', '1.7781', '1.7404', '1.7076', '1.2041', '1.7993', '1.7482', '1.8261', '1.8692']
V23 (numeric, 183 distinct): ['0.9542', '1.0792', '0.9031', '1.0', '1.0414', '1.1139', '0.8451', '1.1461', '1.2305', '0.7782']
V24 (numeric, 217 distinct): ['1.0792', '1.0414', '1.0', '1.1461', '1.1139', '1.1761', '0.9542', '0.9031', '1.301', '1.2553']
V25 (numeric, 918 distinct): ['0.0', '0.3333', '0.6667', '-0.2', '0.5', '0.2', '0.25', '0.1818', '-0.5', '0.1111']
V26 (numeric, 1522 distinct): ['-0.1851', '-0.1903', '-0.189', '-0.1797', '-0.112', '-0.1805', '-0.1865', '-0.1078', '-0.0935', '-0.0481']
V27 (numeric, 388 distinct): ['1.0', '0.1773', '0.2288', '0.2173', '0.1954', '0.2432', '0.2901', '0.2051', '0.9999', '0.3068']
'''

CONTEXT = "Steel Plates Fault Type Classification"
TARGET = CuratedTarget(raw_name='target', new_name="Fault Type", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name='V1', new_name="X Minimum"),
            CuratedFeature(raw_name='V2', new_name="X Maximum"),
            CuratedFeature(raw_name='V3', new_name="Y Minimum"),
            CuratedFeature(raw_name='V4', new_name="Y Maximum"),
            CuratedFeature(raw_name='V5', new_name="Pixels Area"),
            CuratedFeature(raw_name='V6', new_name="X Perimeter"),
            CuratedFeature(raw_name='V7', new_name="Y Perimeter"),
            CuratedFeature(raw_name='V8', new_name="Sum of Luminosity"),
            CuratedFeature(raw_name='V9', new_name="Minimum of Luminosity"),
            CuratedFeature(raw_name='V10', new_name="Maximum of Luminosity"),
            CuratedFeature(raw_name='V11', new_name="Length of Conveyer"),
            CuratedFeature(raw_name='V12', new_name="Type of Steel A300", value_mapping={'0': 'No', '1': 'Yes'}),
            CuratedFeature(raw_name='V13', new_name="Type of Steel A400", value_mapping={'0': 'No', '1': 'Yes'}),
            CuratedFeature(raw_name='V14', new_name="Steel Plate Thickness"),
            CuratedFeature(raw_name='V15', new_name="Edges Index"),
            CuratedFeature(raw_name='V16', new_name="Empty Index"),
            CuratedFeature(raw_name='V17', new_name="Square Index"),
            CuratedFeature(raw_name='V18', new_name="Outside X Index"),
            CuratedFeature(raw_name='V19', new_name="Edges X Index"),
            CuratedFeature(raw_name='V20', new_name="Edges Y Index"),
            CuratedFeature(raw_name='V21', new_name="Outside Global Index",
                           value_mapping={'0.0': 'No', '0.5': 'Partial', '1.0': 'Yes'}),
            CuratedFeature(raw_name='V22', new_name="Log of Areas"),
            CuratedFeature(raw_name='V23', new_name="Log X Index"),
            CuratedFeature(raw_name='V24', new_name="Log Y Index"),
            CuratedFeature(raw_name='V25', new_name="Orientation Index"),
            CuratedFeature(raw_name='V26', new_name="Luminosity Index"),
            CuratedFeature(raw_name='V27', new_name="Sigmoid of Areas")
            ]
