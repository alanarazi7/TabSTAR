from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''Dataset Name: tecator
====
Examples: 240
====
URL: https://www.openml.org/search?type=data&id=505
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

This is the Tecator data set: The task is to predict the fat content of a
meat sample on the basis of its near infrared absorbance spectrum.
1. Statement of permission from Tecator (the original data source)

These data are recorded on a Tecator Infratec Food and Feed Analyzer
working in the wavelength range 850 - 1050 nm by the Near Infrared
Transmission (NIT) principle. Each sample contains finely chopped pure
meat with different moisture, fat and protein contents.

If results from these data are used in a publication we want you to
mention the instrument and company name (Tecator) in the publication.
In addition, please send a preprint of your article to

Karin Thente, Tecator AB,
Box 70, S-263 21 Hoganas, Sweden

The data are available in the public domain with no responsability from
the original data source. The data can be redistributed as long as this
permission note is attached.
For more information about the instrument - call Perstorp Analytical's
representative in your area.


2. Description of the data file

For each meat sample the data consists of a 100 channel spectrum of
absorbances and the contents of moisture (water), fat and protein.
The absorbance is -log10 of the transmittance
measured by the spectrometer. The three contents, measured in percent,
are determined by analytic chemistry.

There are 240 samples which are divided into 5 data sets for the purpose
of model validation and extrapolation studies. The data sets, further
described in reference 1, are:

Data set  Use               Samples
C         Traning               129
M         Monitoring             43
T         Testing                43
E1        Extrapolation, Fat      8
E2        Extrapolation, Protein 17

The data for all 240 samples appear at the end of this file - 25 lines
per sample. The data sets appear in the order of the table above.
The spectra are preprocessed using a principal component analysis on the
data set C, and the first 22 principal components (scaled to unit
variance) are included for each sample.
Thus if you want to use the data for a standard (interpolation) test
of your algorithm, use sample 1-172 for training and sample 173-215
for testing (and ignore the last 25 samples), and use the first 13 or so
principal components to predict the fat content.

Each line contains the 100 absorbances followed by the 22 principal
components and finally the contents of moisture, fat and protein.

Preceeding the data lines, the following lines appear:

real_in=122
real_out=3
training_examples=172
test_examples=43
extrapolation_examples=25


3. More details on how to use the data

The data are made available as a benchmark for regression models. In order
to compare models, it is practical to use the data set as follows:

C and M combined are used to tune (estimate, train) the model. (Some
approaches set aside some training data to control overfitting. These data
should be a subset of C+M. In (1) the subset M was used for this purpose.)

T is used to test the model once it has been tuned.
If each model has an element of randomness (as is the case
for neural networks) the most reliable measure of performance of a single
model is obtained by selecting a handful of models on the basis of C+M and
quoting the average of the performances on T.
In the presence of randomness it is bad practice to train a lot of models
on C+M and then select the best of these on the basis of T.

C, M and T are drawn from the same pool of data, so T is used to test the
ability of the models to interpolate. The data sets E1 and E2 contain
more fat and protein respectively and are intended to be used to test the
ability of the models to extrapolate.


4. Performance of neural network models

The performance is measured as Standard Error of Prediction (SEP) which
is the root mean square of the difference between the true and the predicted
content.

For the prediction of fat on the data set T the following results were obtained

Reference SEP   method (see the papers for details)
(1)       0.65  10-6-1 network, early stopping
(2)       0.52  10-3-1 network, Bayesian
(3)       0.36  13-X-1 network, Bayesian, Automatic Relevance Determination

A linear model with 10 inputs yields SEP=2.78.

5. References

(1) C.Borggaard and H.H.Thodberg,
"Optimal Minimal Neural Interpretation of Spectra",
Analytical Chemistry 64 (1992), p 545-551.
(2) H.H.Thodberg, "Ace of Bayes: Application of Neural Networks with Pruning"
Manuscript 1132, Danish Meat Research Institute (1993),
available by anonymous ftp in the file:
pub/neuroprose/thodberg.ace-of-bayes.ps.Z on the Internet node
archive.cis.ohio-state.edu (128.146.8.52).

(3) Revised and extended version of (2), in preparation, to be
submitted to IEEE Trans. Neural Networks (1995)
available by anonymous ftp in the file:
pub/neuroprose/thodberg.bayesARD.ps.Z on the Internet node
archive.cis.ohio-state.edu (128.146.8.52).

Hans Henrik Thodberg                Email: thodberg@nn.dmri.dk
Danish Meat Research Institute      Phone: (+45) 42 36 12 00
Maglegaardsvej 2, Postboks 57       Fax:   (+45) 42 36 48 36
DK-4000 Roskilde, Denmark

real_in=122
real_out=3
training_examples=172
test_examples=43
extrapolation_examples=25


Note: all 240 samples are included in the same order as mentioned above


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific
====
Target Variable: fat (numeric, 157 distinct): ['6.4', '6.8', '5.6', '11.0', '6.6', '7.7', '10.1', '11.2', '7.2', '31.5']
====
Features:

absorbance_1 (numeric, 216 distinct): ['2.6027', '3.352', '2.8906', '2.677', '2.2863', '2.2408', '2.4869', '2.418', '3.0981', '3.3229']
absorbance_2 (numeric, 216 distinct): ['2.6033', '3.3607', '2.8924', '2.6787', '2.2869', '2.2406', '2.4818', '2.4133', '3.1029', '3.3299']
absorbance_3 (numeric, 216 distinct): ['2.6039', '3.3691', '2.8944', '2.6805', '2.2875', '2.2405', '2.4769', '2.4087', '3.1077', '3.3367']
absorbance_4 (numeric, 214 distinct): ['2.6045', '3.3775', '2.8966', '2.6822', '2.288', '2.2403', '2.4721', '2.4042', '3.1126', '3.3435']
absorbance_5 (numeric, 216 distinct): ['2.6051', '3.3857', '2.8991', '2.684', '2.2886', '2.2403', '2.4675', '2.3998', '3.1176', '3.3504']
absorbance_6 (numeric, 216 distinct): ['2.6058', '3.3939', '2.9021', '2.6859', '2.2894', '2.2403', '2.4631', '2.3956', '3.1228', '3.3575']
absorbance_7 (numeric, 216 distinct): ['2.6067', '3.402', '2.9057', '2.6879', '2.2902', '2.2405', '2.4589', '2.3917', '3.1281', '3.3648']
absorbance_8 (numeric, 216 distinct): ['2.6076', '3.4101', '2.9098', '2.6899', '2.2911', '2.2407', '2.4551', '2.3882', '3.1338', '3.3722']
absorbance_9 (numeric, 216 distinct): ['2.6087', '3.4181', '2.9147', '2.6922', '2.2923', '2.2412', '2.4516', '2.3849', '3.1399', '3.3799']
absorbance_10 (numeric, 216 distinct): ['2.61', '3.4261', '2.9203', '2.6947', '2.2936', '2.2419', '2.4484', '2.3819', '3.1464', '3.388']
absorbance_11 (numeric, 216 distinct): ['2.6115', '3.4341', '2.9267', '2.6973', '2.295', '2.2427', '2.4455', '2.3793', '3.1532', '3.3964']
absorbance_12 (numeric, 215 distinct): ['2.6131', '2.1932', '2.9339', '2.7001', '2.4429', '2.2967', '2.2438', '2.377', '3.1604', '3.405']
absorbance_13 (numeric, 216 distinct): ['2.6149', '3.4501', '2.9421', '2.7029', '2.2985', '2.2451', '2.4408', '2.3751', '3.168', '3.414']
absorbance_14 (numeric, 216 distinct): ['2.6168', '3.4581', '2.9516', '2.706', '2.3004', '2.2466', '2.439', '2.3736', '3.1762', '3.4235']
absorbance_15 (numeric, 216 distinct): ['2.6189', '3.466', '2.9625', '2.7092', '2.3026', '2.2482', '2.4376', '2.3724', '3.1849', '3.4336']
absorbance_16 (numeric, 216 distinct): ['2.6212', '3.474', '2.9747', '2.7126', '2.305', '2.2501', '2.4365', '2.3716', '3.1944', '3.4442']
absorbance_17 (numeric, 216 distinct): ['2.6238', '3.4819', '2.9884', '2.7162', '2.3076', '2.2523', '2.4358', '2.3712', '3.2044', '3.4552']
absorbance_18 (numeric, 216 distinct): ['2.6267', '3.4899', '3.0035', '2.7201', '2.3105', '2.2547', '2.4355', '2.3713', '3.215', '3.4667']
absorbance_19 (numeric, 215 distinct): ['2.6299', '3.498', '2.2483', '2.7357', '2.8122', '3.6331', '3.4793', '3.2265', '2.2575', '2.3136']
absorbance_20 (numeric, 216 distinct): ['2.6333', '3.5061', '3.038', '2.7285', '2.3171', '2.2606', '2.4363', '2.3727', '3.2388', '3.4926']
absorbance_21 (numeric, 215 distinct): ['2.6369', '3.5143', '3.0564', '2.733', '2.3207', '2.2639', '2.4372', '2.3738', '3.2519', '3.5063']
absorbance_22 (numeric, 216 distinct): ['2.6405', '3.5225', '3.0744', '2.7376', '2.3244', '2.2674', '2.4383', '2.3753', '3.2655', '3.5204']
absorbance_23 (numeric, 216 distinct): ['2.6442', '3.5307', '3.0913', '2.7422', '2.3282', '2.271', '2.4396', '2.3769', '3.279', '3.5345']
absorbance_24 (numeric, 216 distinct): ['2.6477', '3.5387', '3.1065', '2.7467', '2.3319', '2.2745', '2.441', '2.3786', '3.2922', '3.5484']
absorbance_25 (numeric, 215 distinct): ['2.6512', '3.5466', '3.1205', '2.274', '2.7512', '2.3355', '2.278', '2.4425', '2.3803', '3.3049']
absorbance_26 (numeric, 216 distinct): ['2.6547', '3.5544', '3.1343', '2.7557', '2.339', '2.2815', '2.4441', '2.3822', '3.3172', '3.5748']
absorbance_27 (numeric, 216 distinct): ['2.6582', '3.5621', '3.1494', '2.76', '2.3426', '2.285', '2.4459', '2.3843', '3.3295', '3.5877']
absorbance_28 (numeric, 216 distinct): ['2.6617', '3.5695', '3.1674', '2.7644', '2.3463', '2.2885', '2.4481', '2.387', '3.3426', '3.6014']
absorbance_29 (numeric, 215 distinct): ['2.6653', '2.7232', '3.1893', '2.7688', '2.3501', '2.2921', '2.451', '2.3904', '3.3573', '3.6166']
absorbance_30 (numeric, 216 distinct): ['2.6691', '3.5836', '3.2153', '2.7732', '2.354', '2.2959', '2.4545', '2.3947', '3.374', '3.6335']
absorbance_31 (numeric, 216 distinct): ['2.6731', '3.5904', '3.2446', '2.7776', '2.3579', '2.2998', '2.4587', '2.3997', '3.3926', '3.6519']
absorbance_32 (numeric, 215 distinct): ['2.6772', '2.2383', '3.0946', '3.2756', '2.7822', '2.362', '2.3038', '2.4634', '2.4053', '3.4122']
absorbance_33 (numeric, 216 distinct): ['2.6814', '3.6037', '3.3064', '2.7868', '2.3664', '2.308', '2.4685', '2.4113', '3.4319', '3.6902']
absorbance_34 (numeric, 216 distinct): ['2.6862', '3.6106', '3.3361', '2.7918', '2.3712', '2.3125', '2.4743', '2.418', '3.4511', '3.7092']
absorbance_35 (numeric, 215 distinct): ['2.6916', '2.9983', '3.3651', '2.7973', '2.3768', '2.3177', '2.4812', '2.4257', '3.4701', '3.7281']
absorbance_36 (numeric, 216 distinct): ['2.6983', '3.6259', '3.3948', '2.8039', '2.3836', '2.3239', '2.4898', '2.4351', '3.4899', '3.7478']
absorbance_37 (numeric, 216 distinct): ['2.7066', '3.6351', '3.4266', '2.8118', '2.3921', '2.3316', '2.5007', '2.4467', '3.5113', '3.7692']
absorbance_38 (numeric, 216 distinct): ['2.7167', '3.6455', '3.4595', '2.8213', '2.4024', '2.3413', '2.5138', '2.4603', '3.5347', '3.7925']
absorbance_39 (numeric, 216 distinct): ['2.7286', '3.6572', '3.4913', '2.8321', '2.4143', '2.3528', '2.5287', '2.4757', '3.559', '3.8168']
absorbance_40 (numeric, 215 distinct): ['2.7421', '3.6702', '3.5188', '2.8441', '2.4278', '2.366', '2.5451', '2.4923', '3.5825', '3.8407']
absorbance_41 (numeric, 216 distinct): ['2.7569', '3.6845', '3.5383', '2.8573', '2.4426', '2.3807', '2.5624', '2.5093', '3.6033', '3.8626']
absorbance_42 (numeric, 216 distinct): ['2.7728', '3.7', '3.547', '2.8714', '2.4585', '2.3966', '2.5802', '2.5262', '3.6194', '3.8809']
absorbance_43 (numeric, 216 distinct): ['2.7897', '3.7167', '3.5435', '2.8861', '2.4753', '2.4135', '2.5981', '2.5426', '3.6293', '3.8944']
absorbance_44 (numeric, 216 distinct): ['2.8076', '3.7344', '3.5294', '2.9016', '2.493', '2.4312', '2.616', '2.5584', '3.6327', '3.903']
absorbance_45 (numeric, 216 distinct): ['2.8268', '3.7532', '3.5088', '2.918', '2.5119', '2.4499', '2.6345', '2.5744', '3.6313', '3.9079']
absorbance_46 (numeric, 216 distinct): ['2.8475', '3.7733', '3.4875', '2.9356', '2.5323', '2.4701', '2.6543', '2.5916', '3.6279', '3.9115']
absorbance_47 (numeric, 216 distinct): ['2.8706', '3.7948', '3.471', '2.9552', '2.555', '2.4922', '2.6768', '2.6115', '3.6262', '3.9168']
absorbance_48 (numeric, 216 distinct): ['2.8972', '3.8187', '3.4645', '2.9778', '2.5812', '2.5171', '2.7037', '2.6359', '3.6292', '3.9268']
absorbance_49 (numeric, 216 distinct): ['2.928', '3.8453', '3.4696', '3.0041', '2.6113', '2.546', '2.7357', '2.6656', '3.64', '3.9442']
absorbance_50 (numeric, 216 distinct): ['2.9636', '3.8754', '3.4867', '3.0348', '2.6462', '2.5796', '2.7735', '2.7014', '3.6594', '3.9696']
absorbance_51 (numeric, 216 distinct): ['3.0045', '3.9091', '3.5155', '3.07', '2.6858', '2.6184', '2.8173', '2.7432', '3.6875', '4.0032']
absorbance_52 (numeric, 216 distinct): ['3.05', '3.9462', '3.5542', '3.1092', '2.7296', '2.6622', '2.8662', '2.7901', '3.7237', '4.0444']
absorbance_53 (numeric, 216 distinct): ['3.0981', '3.9853', '3.5994', '3.1507', '2.7756', '2.7094', '2.918', '2.8399', '3.7658', '4.0913']
absorbance_54 (numeric, 216 distinct): ['3.1458', '4.0241', '3.6468', '3.1917', '2.8208', '2.7571', '2.9692', '2.8892', '3.8106', '4.1405']
absorbance_55 (numeric, 215 distinct): ['3.1893', '4.0596', '3.6915', '3.2287', '2.8615', '2.8016', '3.0159', '2.9342', '3.8539', '4.1875']
absorbance_56 (numeric, 215 distinct): ['3.7293', '3.2256', '4.0896', '3.259', '2.8951', '2.8396', '3.055', '2.9718', '3.8917', '4.2285']
absorbance_57 (numeric, 215 distinct): ['3.2532', '4.1127', '3.7585', '3.2815', '2.9204', '2.8691', '3.0849', '3.0007', '3.1836', '3.9218']
absorbance_58 (numeric, 216 distinct): ['3.2726', '4.1294', '3.7792', '3.2965', '2.9382', '2.8905', '3.1065', '3.0217', '3.9439', '4.2858']
absorbance_59 (numeric, 215 distinct): ['3.2858', '3.9592', '3.1594', '2.8237', '3.5491', '3.549', '4.3194', '4.3033', '3.281', '4.1409']
absorbance_60 (numeric, 216 distinct): ['3.2946', '4.1487', '3.8031', '3.3112', '2.9584', '2.9151', '3.1321', '3.0466', '3.9696', '4.3154']
absorbance_61 (numeric, 216 distinct): ['3.3005', '4.1542', '3.8098', '3.314', '2.9639', '2.9219', '3.1396', '3.054', '3.9767', '4.3239']
absorbance_62 (numeric, 216 distinct): ['3.3046', '4.1581', '3.8149', '3.3148', '2.9675', '2.9263', '3.145', '3.0593', '3.9815', '4.3301']
absorbance_63 (numeric, 216 distinct): ['3.3069', '4.1606', '3.8182', '3.3141', '2.9696', '2.9291', '3.1486', '3.0629', '3.9846', '4.3346']
absorbance_64 (numeric, 216 distinct): ['3.3074', '4.1616', '3.8199', '3.3118', '2.9701', '2.9301', '3.1501', '3.0646', '3.9861', '4.3377']
absorbance_65 (numeric, 216 distinct): ['3.3062', '4.1609', '3.8198', '3.3077', '2.9689', '2.9294', '3.1497', '3.0644', '3.9859', '4.3387']
absorbance_66 (numeric, 216 distinct): ['3.3029', '4.1585', '3.8178', '3.3018', '2.9659', '2.9267', '3.1473', '3.0621', '3.9837', '4.3376']
absorbance_67 (numeric, 216 distinct): ['3.2977', '4.1545', '3.814', '3.294', '2.961', '2.9221', '3.1427', '3.0577', '3.9797', '4.3345']
absorbance_68 (numeric, 215 distinct): ['3.2906', '3.2443', '3.8085', '3.2845', '2.9544', '2.9157', '3.1361', '3.0515', '3.974', '4.3296']
absorbance_69 (numeric, 216 distinct): ['3.2819', '4.1417', '3.8016', '3.2734', '2.9462', '2.9076', '3.1277', '3.0435', '3.9667', '4.3232']
absorbance_70 (numeric, 216 distinct): ['3.2716', '4.1333', '3.7936', '3.2609', '2.9366', '2.8978', '3.1176', '3.0339', '3.9579', '4.3153']
absorbance_71 (numeric, 216 distinct): ['3.2598', '4.1237', '3.7847', '3.247', '2.9257', '2.8867', '3.106', '3.0228', '3.948', '4.3059']
absorbance_72 (numeric, 216 distinct): ['3.2468', '4.113', '3.7751', '3.2319', '2.9137', '2.8743', '3.093', '3.0104', '3.9369', '4.2954']
absorbance_73 (numeric, 216 distinct): ['3.2325', '4.1012', '3.7647', '3.2157', '2.9005', '2.8608', '3.0789', '2.9969', '3.9249', '4.2841']
absorbance_74 (numeric, 216 distinct): ['3.2173', '4.0884', '3.7537', '3.1986', '2.8865', '2.8462', '3.0636', '2.9822', '3.912', '4.2721']
absorbance_75 (numeric, 216 distinct): ['3.2012', '4.0749', '3.742', '3.1806', '2.8716', '2.8307', '3.0472', '2.9666', '3.8984', '4.259']
absorbance_76 (numeric, 216 distinct): ['3.1842', '4.0608', '3.7297', '3.1618', '2.856', '2.8143', '3.03', '2.9501', '3.884', '4.2452']
absorbance_77 (numeric, 216 distinct): ['3.1665', '4.0463', '3.717', '3.1425', '2.8398', '2.7973', '3.012', '2.9329', '3.8689', '4.2306']
absorbance_78 (numeric, 216 distinct): ['3.1484', '4.0312', '3.7044', '3.1227', '2.8232', '2.7798', '2.9936', '2.9152', '3.8534', '4.2158']
absorbance_79 (numeric, 216 distinct): ['3.1299', '4.0158', '3.6921', '3.1025', '2.8064', '2.7619', '2.9747', '2.8971', '3.838', '4.2011']
absorbance_80 (numeric, 216 distinct): ['3.1111', '4.0', '3.6804', '3.0821', '2.7892', '2.7437', '2.9555', '2.8787', '3.8228', '4.1867']
absorbance_81 (numeric, 215 distinct): ['2.7301', '3.0919', '3.9841', '3.669', '3.0613', '2.7718', '2.7251', '2.9359', '2.8599', '3.8077']
absorbance_82 (numeric, 216 distinct): ['3.0724', '3.9678', '3.6576', '3.0402', '2.754', '2.7063', '2.9159', '2.8408', '3.7925', '4.1573']
absorbance_83 (numeric, 216 distinct): ['3.0524', '3.9511', '3.6457', '3.0187', '2.7358', '2.6871', '2.8955', '2.8213', '3.7771', '4.1424']
absorbance_84 (numeric, 216 distinct): ['3.032', '3.9341', '3.6327', '2.9968', '2.7173', '2.6674', '2.8746', '2.8013', '3.761', '4.1271']
absorbance_85 (numeric, 216 distinct): ['3.0112', '3.9168', '3.6181', '2.9744', '2.6984', '2.6473', '2.8532', '2.7808', '3.7439', '4.1108']
absorbance_86 (numeric, 215 distinct): ['2.99', '3.8993', '3.6013', '2.9517', '2.6791', '2.6268', '2.8313', '2.7597', '3.7253', '4.0932']
absorbance_87 (numeric, 216 distinct): ['2.9685', '3.8815', '3.5826', '2.9287', '2.6595', '2.6058', '2.8091', '2.7382', '3.7052', '4.0742']
absorbance_88 (numeric, 216 distinct): ['2.9469', '3.8636', '3.5629', '2.9057', '2.64', '2.5847', '2.7869', '2.7167', '3.6843', '4.0546']
absorbance_89 (numeric, 216 distinct): ['2.9255', '3.8459', '3.5435', '2.8828', '2.6206', '2.5638', '2.7649', '2.6955', '3.6635', '4.0354']
absorbance_90 (numeric, 216 distinct): ['2.9043', '3.8283', '3.5254', '2.8602', '2.6014', '2.5431', '2.7434', '2.6747', '3.6433', '4.0169']
absorbance_91 (numeric, 216 distinct): ['2.8835', '3.8109', '3.5088', '2.838', '2.5826', '2.5227', '2.7223', '2.6546', '3.624', '3.9992']
absorbance_92 (numeric, 216 distinct): ['2.863', '3.7936', '3.4931', '2.816', '2.5641', '2.5027', '2.7019', '2.6349', '3.6055', '3.9818']
absorbance_93 (numeric, 215 distinct): ['2.8429', '2.4829', '3.4778', '2.7944', '2.546', '2.483', '2.6819', '2.6157', '3.5873', '3.9647']
absorbance_94 (numeric, 214 distinct): ['3.0144', '2.8232', '2.7833', '3.5694', '3.9479', '3.921', '3.0896', '2.597', '2.6623', '2.4403']
absorbance_95 (numeric, 216 distinct): ['2.804', '3.7433', '3.4463', '2.7525', '2.511', '2.4447', '2.6433', '2.5789', '3.5513', '3.9313']
absorbance_96 (numeric, 216 distinct): ['2.7853', '3.7272', '3.4295', '2.7323', '2.4943', '2.4263', '2.6249', '2.5612', '3.5332', '3.9146']
absorbance_97 (numeric, 216 distinct): ['2.7673', '3.7115', '3.4114', '2.7127', '2.4782', '2.4087', '2.6071', '2.5441', '3.5146', '3.8976']
absorbance_98 (numeric, 216 distinct): ['2.75', '3.6964', '3.3915', '2.6938', '2.4627', '2.3916', '2.5899', '2.5272', '3.4954', '3.8803']
absorbance_99 (numeric, 216 distinct): ['2.7332', '3.6817', '3.3696', '2.6754', '2.4477', '2.3752', '2.573', '2.5107', '3.4752', '3.8624']
absorbance_100 (numeric, 216 distinct): ['2.717', '3.6672', '3.3462', '2.6574', '2.4332', '2.3593', '2.5567', '2.4946', '3.4541', '3.8439']
principal_component_1 (numeric, 216 distinct): ['-0.6001', '1.1574', '0.4653', '-0.5288', '-1.2302', '-1.3375', '-0.9442', '-1.0826', '0.7694', '1.3874']
principal_component_2 (numeric, 216 distinct): ['-0.0855', '0.975', '-0.3593', '1.0525', '-0.5026', '-0.6422', '-0.5839', '-0.5485', '0.2222', '-0.3883']
principal_component_3 (numeric, 217 distinct): ['0.2724', '-1.2269', '0.3415', '0.0181', '-0.255', '-0.4269', '-0.147', '-0.1054', '0.6907', '0.1296']
principal_component_4 (numeric, 217 distinct): ['1.153', '2.0155', '1.3859', '0.8124', '1.1452', '0.4935', '0.5958', '0.8419', '2.4808', '0.0113']
principal_component_5 (numeric, 216 distinct): ['0.7304', '-0.7052', '0.9908', '0.2583', '-1.0589', '-0.0152', '3.0301', '2.8388', '1.0771', '-0.0378']
principal_component_6 (numeric, 216 distinct): ['0.6791', '0.3427', '-0.9103', '-0.3753', '0.5256', '-0.7721', '3.4824', '3.1185', '-1.1744', '0.2243']
principal_component_7 (numeric, 216 distinct): ['0.2042', '0.0141', '0.5747', '0.4752', '0.2971', '-0.5932', '0.7161', '0.6988', '-1.0989', '-1.5037']
principal_component_8 (numeric, 216 distinct): ['-0.9746', '-0.6131', '-0.3673', '-0.017', '0.2704', '-0.8277', '1.9925', '2.2591', '2.4127', '1.4477']
principal_component_9 (numeric, 216 distinct): ['-0.9', '-0.671', '0.3483', '-0.0935', '-0.6622', '-0.4636', '0.7081', '0.988', '-1.1208', '-1.0257']
principal_component_10 (numeric, 216 distinct): ['0.2019', '-0.6549', '-1.4403', '-0.4967', '0.767', '1.6208', '1.1606', '0.378', '-0.4423', '0.4453']
principal_component_11 (numeric, 216 distinct): ['0.1056', '1.3115', '1.7842', '-0.457', '0.3726', '-0.3111', '-0.3878', '-1.2562', '-1.4303', '0.2463']
principal_component_12 (numeric, 216 distinct): ['0.266', '1.6625', '1.1533', '0.6699', '-0.4807', '-0.5669', '1.5815', '1.6498', '-1.5917', '-1.5001']
principal_component_13 (numeric, 216 distinct): ['-0.3882', '-0.3437', '1.5856', '0.7282', '0.9281', '-0.2317', '0.5124', '-0.2167', '-0.1668', '-1.42']
principal_component_14 (numeric, 217 distinct): ['0.1219', '0.3823', '-0.7859', '-1.1388', '-0.8284', '0.3308', '-0.22', '1.2069', '0.7215', '1.1089']
principal_component_15 (numeric, 216 distinct): ['0.5354', '-1.1151', '-0.7755', '0.3912', '1.0117', '1.4107', '0.9534', '0.947', '-0.7871', '-1.2247']
principal_component_16 (numeric, 216 distinct): ['0.4486', '-0.7463', '-0.1516', '0.426', '1.0898', '1.8062', '1.2749', '1.1923', '-0.6101', '-1.3317']
principal_component_17 (numeric, 216 distinct): ['0.4041', '-1.4704', '0.2004', '0.3778', '1.1867', '1.1977', '0.5713', '0.6825', '-0.515', '-0.9566']
principal_component_18 (numeric, 217 distinct): ['-1.1098', '-0.3391', '-0.1876', '1.269', '-0.2332', '-1.9482', '-1.2303', '-0.7696', '0.0851', '-0.3672']
principal_component_19 (numeric, 216 distinct): ['0.3843', '-1.2551', '0.3073', '0.422', '0.9771', '1.9518', '0.5', '0.838', '0.0408', '-0.3706']
principal_component_20 (numeric, 216 distinct): ['-0.6213', '1.151', '0.5793', '-0.5367', '-1.2278', '-1.2471', '-0.7134', '-0.8601', '0.8713', '1.4178']
principal_component_21 (numeric, 216 distinct): ['-0.5795', '1.0337', '0.7333', '-0.4729', '-0.9454', '-1.3416', '-1.0465', '-1.0304', '0.9802', '1.0514']
principal_component_22 (numeric, 216 distinct): ['-0.6196', '1.1139', '0.4662', '-0.519', '-1.2076', '-1.3462', '-0.9382', '-1.0706', '0.7814', '1.3864']
moisture (numeric, 141 distinct): ['72.5', '50.3', '73.5', '69.3', '61.4', '72.7', '63.6', '71.6', '73.4', '70.3']
protein (numeric, 97 distinct): ['19.2', '20.5', '15.2', '19.5', '19.3', '19.6', '20.1', '20.7', '21.6', '19.7']
'''

CONTEXT = "Fat Content of Meat"
TARGET = CuratedTarget(raw_name="fat", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []