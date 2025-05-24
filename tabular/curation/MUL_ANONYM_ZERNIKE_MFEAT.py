from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mfeat-zernike
====
Examples: 2000
====
URL: https://www.openml.org/search?type=data&id=22
====
Description: **Author**: Robert P.W. Duin, Department of Applied Physics, Delft University of Technology  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Multiple+Features) - 1998  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)   

**Multiple Features Dataset: Zernike**  
One of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps. Corresponding patterns in different datasets correspond to the same original character. 200 instances per class (for a total of 2,000 instances) have been digitized in binary images. 

In this dataset, these digits are represented in terms of 47 Zernike moments. 

### Attribute Information  
The attributes represent 47 rotation invariant Zernike moments. They can't distinguish samples of class '6' from those of class '9'. More information on Zernike moments can be found in:  
A. Khotanzad and Y.H. Hong: Rotation invariant pattern recognition using Zernike moments. Int. Conf. on Pattern Recognition, Rome 1998, pp. 326-328.

### Relevant Papers  
A slightly different version of the database is used in  
M. van Breukelen, R.P.W. Duin, D.M.J. Tax, and J.E. den Hartog, Handwritten digit recognition by combined classifiers, Kybernetika, vol. 34, no. 4, 1998, 381-386.
 
The database as is is used in:  
A.K. Jain, R.P.W. Duin, J. Mao, Statistical Pattern Recognition: A Review, IEEE Transactions on Pattern Analysis and Machine Intelligence archive, Volume 22 Issue 1, January 2000
====
Target Variable: class (nominal, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
====
Features:

att1 (numeric, 1987 distinct): ['0.0666', '0.0297', '0.0457', '0.0297', '0.0454', '0.0201', '0.1756', '0.0341', '0.0606', '0.0433']
att2 (numeric, 1994 distinct): ['1.7223', '2.8549', '0.7768', '0.7768', '0.3805', '2.4191', '0.8315', '1.1205', '0.9726', '1.3404']
att3 (numeric, 1994 distinct): ['19.1391', '52.6759', '9.8299', '9.8299', '32.5646', '33.2039', '15.3518', '13.8381', '8.8332', '20.3597']
att4 (numeric, 1994 distinct): ['79.6434', '146.6512', '40.5871', '40.5871', '72.9324', '95.2416', '75.8066', '72.7273', '20.4108', '85.3526']
att5 (numeric, 1994 distinct): ['75.9188', '197.3063', '51.1881', '51.1881', '96.9856', '74.2478', '171.5542', '54.4046', '72.544', '125.9155']
att6 (numeric, 1994 distinct): ['334.5819', '169.6858', '148.0208', '148.0209', '301.1837', '418.2944', '490.1566', '224.4711', '344.4905', '403.3029']
att7 (numeric, 1994 distinct): ['55.5158', '309.8766', '124.9393', '124.9392', '0.2581', '11.5879', '206.416', '26.8236', '10.3447', '216.7255']
att8 (numeric, 1992 distinct): ['0.057', '0.0651', '0.1858', '0.0833', '0.0912', '0.068', '0.2112', '0.0833', '0.1221', '0.1563']
att9 (numeric, 1994 distinct): ['1.4398', '2.2875', '2.7804', '2.7804', '2.9762', '2.9205', '2.6016', '2.699', '2.6435', '4.3192']
att10 (numeric, 1994 distinct): ['14.7268', '17.4943', '30.029', '30.029', '39.4931', '24.6439', '11.4727', '22.795', '32.586', '15.6085']
att11 (numeric, 1994 distinct): ['71.0955', '77.8962', '119.1603', '119.1603', '49.1858', '33.2272', '20.0434', '81.1592', '123.516', '75.1171']
att12 (numeric, 1994 distinct): ['222.9755', '32.3119', '186.7191', '186.7191', '92.0045', '36.7597', '110.8683', '179.8061', '182.0892', '83.9299']
att13 (numeric, 1994 distinct): ['245.5424', '115.9167', '153.3884', '153.3884', '279.0298', '237.2704', '99.0253', '183.8093', '159.4583', '59.5515']
att14 (numeric, 1993 distinct): ['0.1949', '0.3479', '0.2718', '0.0282', '0.0934', '0.3017', '0.0934', '0.1657', '0.1357', '0.1066']
att15 (numeric, 1994 distinct): ['5.2477', '15.6826', '2.7186', '2.7186', '9.5418', '9.8768', '4.4658', '3.9854', '2.3868', '6.1526']
att16 (numeric, 1994 distinct): ['43.0016', '89.0939', '19.8948', '19.8948', '47.3299', '54.3579', '37.2037', '40.428', '12.5298', '48.6183']
att17 (numeric, 1994 distinct): ['113.3941', '196.7459', '34.9368', '34.9368', '86.9003', '52.3623', '71.6827', '103.7578', '65.4988', '77.2377']
att18 (numeric, 1994 distinct): ['139.847', '185.6441', '25.5746', '25.5747', '126.5876', '289.7631', '303.833', '111.3103', '172.1431', '221.2354']
att19 (numeric, 1994 distinct): ['94.291', '190.3023', '37.597', '37.5969', '139.1936', '127.6847', '444.8187', '111.4427', '151.4579', '0.9804']
att20 (numeric, 1991 distinct): ['0.1771', '0.2898', '0.2618', '0.3787', '0.2295', '0.3455', '0.3455', '0.4042', '0.3217', '0.2482']
att21 (numeric, 1994 distinct): ['4.2072', '5.3582', '8.7103', '8.7103', '13.89', '8.9445', '2.8082', '6.645', '9.4798', '4.2543']
att22 (numeric, 1994 distinct): ['35.8347', '52.1236', '56.2697', '56.2697', '16.5856', '22.9561', '7.5055', '41.6693', '66.5308', '48.584']
att23 (numeric, 1994 distinct): ['142.3974', '63.4996', '86.9177', '86.9177', '81.1605', '53.3435', '14.1993', '96.3923', '128.8508', '136.7903']
att24 (numeric, 1994 distinct): ['164.5858', '140.6666', '157.3308', '157.3308', '203.2688', '174.7238', '109.8205', '142.235', '153.6917', '109.1444']
att25 (numeric, 1993 distinct): ['0.7015', '2.2345', '0.5937', '1.344', '0.3671', '0.3671', '1.4053', '0.8813', '0.3158', '0.5526']
att26 (numeric, 1994 distinct): ['14.5642', '32.9583', '6.2982', '6.2982', '18.2167', '19.422', '11.9026', '13.9994', '4.6324', '17.212']
att27 (numeric, 1994 distinct): ['88.8448', '135.8451', '40.8237', '40.8237', '60.0158', '54.8292', '13.9929', '88.0525', '54.3716', '54.8336']
att28 (numeric, 1993 distinct): ['96.4126', '230.4948', '105.7474', '179.0731', '146.8402', '146.8402', '302.4758', '94.8792', '107.0069', '192.2693']
att29 (numeric, 1994 distinct): ['84.1597', '19.9397', '15.2421', '15.2421', '223.3694', '206.4338', '484.7849', '156.8767', '130.062', '123.6938']
att30 (numeric, 1994 distinct): ['0.5947', '0.8005', '1.242', '1.242', '2.2591', '1.484', '0.3447', '0.9487', '1.3516', '0.5859']
att31 (numeric, 1994 distinct): ['11.9891', '21.0967', '17.8145', '17.8145', '9.7647', '10.4621', '6.8366', '13.96', '23.2398', '19.5515']
att32 (numeric, 1994 distinct): ['69.7969', '81.9697', '17.2216', '17.2217', '56.4702', '38.7526', '41.1729', '32.5048', '71.0507', '132.3413']
att33 (numeric, 1994 distinct): ['58.382', '53.7662', '134.2987', '134.2987', '135.1483', '96.543', '67.5832', '85.0205', '141.2208', '171.3636']
att34 (numeric, 1994 distinct): ['2.3748', '5.7361', '0.9796', '0.9796', '3.2528', '3.3127', '1.8756', '2.3199', '0.7981', '2.9061']
att35 (numeric, 1994 distinct): ['40.3935', '59.0674', '21.9899', '21.9899', '25.3021', '29.8283', '2.2804', '41.707', '25.9438', '26.0179']
att36 (numeric, 1994 distinct): ['183.5201', '348.8076', '215.9587', '215.9587', '291.8989', '305.8418', '170.2658', '251.5348', '208.0789', '249.964']
att37 (numeric, 1994 distinct): ['151.5124', '45.2708', '112.3251', '112.3251', '36.3077', '1.3782', '158.8184', '33.2139', '123.4525', '46.2004']
att38 (numeric, 1994 distinct): ['2.0063', '4.0124', '2.85', '2.85', '2.4509', '2.1863', '1.7135', '2.3212', '3.9944', '3.7218']
att39 (numeric, 1994 distinct): ['24.1728', '49.5745', '6.1172', '6.1172', '26.8723', '16.2499', '33.8103', '5.2842', '27.4943', '72.6118']
att40 (numeric, 1994 distinct): ['99.2772', '70.4077', '182.7169', '182.7169', '84.0913', '48.5609', '9.8589', '163.4221', '154.2917', '184.5259']
att41 (numeric, 1993 distinct): ['8.5116', '12.2703', '5.0313', '5.0315', '5.0313', '6.9544', '8.3169', '5.6993', '2.6389', '5.6097']
att42 (numeric, 1994 distinct): ['131.2319', '229.5614', '148.4477', '148.4477', '207.5678', '219.3408', '148.1381', '169.7016', '141.4625', '181.8274']
att43 (numeric, 1994 distinct): ['440.5763', '261.4338', '336.8185', '336.8185', '320.7165', '391.8975', '326.2395', '333.0804', '432.6432', '381.1191']
att44 (numeric, 1994 distinct): ['4.4654', '12.8547', '3.1255', '3.1255', '6.283', '3.2309', '9.7111', '0.838', '5.5237', '18.0222']
att45 (numeric, 1994 distinct): ['93.0998', '95.3796', '160.0452', '160.0452', '48.4797', '26.0429', '20.0072', '149.6322', '116.2731', '127.1844']
att46 (numeric, 1994 distinct): ['38.8548', '67.0334', '43.6538', '43.6538', '61.9532', '66.2817', '47.0326', '49.5394', '41.1734', '54.6614']
att47 (numeric, 1994 distinct): ['503.5095', '381.2366', '408.1714', '408.1714', '484.7982', '563.5109', '539.2085', '443.2455', '507.7478', '515.5188']
'''

CONTEXT = "Anonymized Dataset: Zernike Moments of Handwritten Numerals"
TARGET = CuratedTarget(raw_name="class", new_name="Numerals", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []