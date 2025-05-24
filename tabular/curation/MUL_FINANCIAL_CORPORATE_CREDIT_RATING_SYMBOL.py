from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: Corporate-Credit-Rating
====
Examples: 2029
====
URL: https://www.openml.org/search?type=data&id=43344
====
Description: Context
A corporate credit rating expresses the ability of a firm to repay its debt to creditors. Credit rating agencies are the entities responsible to make the assessment and give a verdict.  When a big corporation from the US or anywhere in the world wants to issue a new bond it hires a credit agency to make an assessment so that investors can know how trustworthy is the company. The assessment is based especially in the financials indicators that come from the balance sheet. Some of the most important agencies in the world are Moodys, Fitch and Standard and Poors. 
Content
A list of 2029 credit ratings issued by major agencies such as Standard and Poors to big US firms (traded on NYSE or Nasdaq) from 2010 to 2016. 
There are 30 features for every company of which 25 are financial indicators. They can be divided in:

Liquidity Measurement Ratios: currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding
Profitability Indicator Ratios: grossProfitMargin, operatingProfitMargin, pretaxProfitMargin, netProfitMargin, effectiveTaxRate, returnOnAssets, returnOnEquity, returnOnCapitalEmployed
Debt Ratios: debtRatio, debtEquityRatio
Operating Performance Ratios: assetTurnover
Cash Flow Indicator Ratios: operatingCashFlowPerShare, freeCashFlowPerShare, cashPerShare, operatingCashFlowSalesRatio, freeCashFlowOperatingCashFlowRatio

For more information about financial indicators visit: https://financialmodelingprep.com/market-indexes-major-markets
The additional features are Name, Symbol (for trading), Rating Agency Name, Date and Sector. 
The dataset is unbalanced, here is the frequency of ratings:

AAA:         7
AA:            89
A:           398
BBB:           671
BB:           490
B:           302
CCC:       64
CC:           5
C:           2
D:           1

Acknowledgements
This dataset was possible thanks to financialmodelingprep and opendatasoft - the sources of the data. To see how the data was integrated and reshaped check here.
Inspiration
Is it possible to forecast the rating an agency will give to a company based on its financials?
====
Features:

Rating (string, 10 distinct): ['BBB', 'BB', 'A', 'B', 'AA', 'CCC', 'AAA', 'CC', 'C', 'D']
Name (string, 593 distinct): ['BCE, Inc.', 'Walt Disney Company (The)', 'CSX Corporation', 'CoreLogic, Inc.', 'PLDT Inc.', 'Jabil Inc.', 'Bunge Limited', 'Parker-Hannifin Corporation', 'WT Offshore, Inc.', 'Exelon Corporation']
Symbol (string, 593 distinct): ['BCE', 'DIS', 'CSX', 'CLGX', 'PHI', 'JBL', 'BG', 'PH', 'WTI', 'EXC']
Rating_Agency_Name (string, 5 distinct): ["Standard  Poor's Ratings Services", 'Egan-Jones Ratings Company', "Moody's Investors Service", 'Fitch Ratings', 'DBRS']
Date (string, 904 distinct): ['6/15/2012', '11/30/2015', '10/24/2016', '10/28/2015', '12/14/2016', '11/6/2015', '9/30/2015', '9/22/2016', '2/18/2016', '9/10/2015']
Sector (string, 12 distinct): ['Energy', 'Basic Industries', 'Consumer Services', 'Technology', 'Capital Goods', 'Public Utilities', 'Health Care', 'Consumer Non-Durables', 'Consumer Durables', 'Transportation']
currentRatio (numeric, 2029 distinct): ['0.9459', '2.6287', '0.7039', '0.5492', '1.4173', '0.9335', '1.3467', '1.1105', '2.2694', '2.7948']
quickRatio (numeric, 2029 distinct): ['0.4264', '1.3982', '0.4572', '0.354', '0.8838', '0.5736', '0.88', '0.6883', '1.5609', '2.2141']
cashRatio (numeric, 2025 distinct): ['0.0', '0.0997', '0.668', '0.0917', '0.0682', '0.1084', '0.0195', '0.0972', '0.0478', '0.5633']
daysOfSalesOutstanding (numeric, 1836 distinct): ['0.0', '44.2032', '51.7387', '29.0487', '52.0734', '60.1572', '64.97', '64.4088', '70.723', '67.1367']
netProfitMargin (numeric, 2027 distinct): ['0.0412', '0.1188', '0.0375', '0.0163', '-0.0566', '0.1191', '0.0965', '0.0935', '0.074', '0.073']
pretaxProfitMargin (numeric, 2027 distinct): ['0.0665', '0.2083', '0.0494', '0.0253', '-0.0873', '0.1521', '0.1177', '0.1255', '0.0973', '0.0839']
grossProfitMargin (numeric, 1626 distinct): ['1.0', '0.0736', '0.1766', '0.9231', '0.5664', '0.2888', '0.1701', '0.2878', '0.2807', '0.0861']
operatingProfitMargin (numeric, 2027 distinct): ['0.0736', '0.2403', '0.0615', '0.0473', '-0.0873', '0.21', '0.2169', '0.1456', '0.1153', '0.1178']
returnOnAssets (numeric, 2029 distinct): ['0.0412', '-0.0593', '0.0279', '0.0138', '0.0671', '0.0538', '0.0493', '0.0618', '-0.0474', '0.0043']
returnOnCapitalEmployed (numeric, 2029 distinct): ['0.0915', '-0.0517', '0.0393', '0.0183', '0.1078', '0.095', '0.0689', '0.1126', '-0.0675', '0.0117']
returnOnEquity (numeric, 2029 distinct): ['0.1651', '-0.2192', '0.0815', '0.0452', '0.1782', '0.145', '0.1318', '0.1644', '1.0333', '0.0188']
assetTurnover (numeric, 2029 distinct): ['1.0989', '1.0938', '0.2346', '0.1427', '0.7175', '0.7266', '0.6749', '0.7336', '1.1552', '0.9075']
fixedAssetTurnover (numeric, 2029 distinct): ['5.5355', '3.0994', '0.3448', '0.2331', '3.9087', '4.1957', '4.5987', '4.6812', '17.3938', '12.7397']
debtEquityRatio (numeric, 2024 distinct): ['0.0', '3.008', '2.0661', '1.6561', '1.6978', '1.6737', '1.6609', '-22.805', '3.3407', '1.0108']
debtRatio (numeric, 2005 distinct): ['1.0', '0.0', '0.7505', '0.6739', '0.6293', '0.626', '0.6242', '1.0459', '0.7696', '0.5027']
effectiveTaxRate (numeric, 1994 distinct): ['0.0', '0.2549', '0.3546', '0.3333', '0.2308', '0.3769', '0.4116', '0.3492', '0.1433', '0.1801']
freeCashFlowOperatingCashFlowRatio (numeric, 1857 distinct): ['1.0', '0.9381', '0.8165', '0.4376', '0.3131', '0.4384', '-0.3054', '-0.0674', '0.6353', '0.6145']
freeCashFlowPerShare (numeric, 2027 distinct): ['10.1648', '2.7375', '6.8107', '0.6885', '8.3758', '-1.2728', '-0.3163', '4.213', '4.1457', '3.1167']
cashPerShare (numeric, 2029 distinct): ['9.8094', '3.7294', '0.6028', '0.67', '1.1193', '1.1565', '1.6746', '1.3089', '4.6064', '7.8205']
companyEquityMultiplier (numeric, 2029 distinct): ['4.008', '3.6956', '2.9154', '3.2818', '2.6561', '2.6978', '2.6737', '2.6609', '-21.805', '4.3407']
ebitPerRevenue (numeric, 2027 distinct): ['0.0665', '0.2083', '0.0494', '0.0253', '-0.0873', '0.1521', '0.1177', '0.1255', '0.0973', '0.0839']
enterpriseValueMultiple (numeric, 2029 distinct): ['7.0571', '10.4221', '12.7099', '20.6263', '15.1482', '15.84', '15.4848', '13.2654', '25.0633', '14.5982']
operatingCashFlowPerShare (numeric, 2027 distinct): ['10.8358', '3.3527', '15.5654', '2.1993', '8.3758', '4.167', '4.6926', '6.6315', '6.747', '5.2011']
operatingCashFlowSalesRatio (numeric, 2027 distinct): ['0.0677', '0.2085', '0.0586', '0.0539', '0.2587', '0.2476', '0.2755', '0.1475', '0.1476', '0.1177']
payablesTurnover (numeric, 1768 distinct): ['0.0', '3.9067', '3.1788', '1.1173', '0.7815', '2.8629', '4.3253', '4.752', '5.9783', '6.4356']
'''

CONTEXT = "Corporate Credit Rating"
TARGET = CuratedTarget(raw_name='Rating', new_name='Corporate Credit Rating', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name='Name', new_name='Company Name'),
            CuratedFeature(raw_name='Symbol', new_name='Trading Symbol'),
            CuratedFeature(raw_name='Rating_Agency_Name', new_name='Rating Agency Name'),
            CuratedFeature(raw_name='Date', feat_type=FeatureType.DATE),
            CuratedFeature(raw_name='currentRatio', new_name='Current Ratio'),
            CuratedFeature(raw_name='quickRatio', new_name='Quick Ratio'),
            CuratedFeature(raw_name='cashRatio', new_name='Cash Ratio'),
            CuratedFeature(raw_name='daysOfSalesOutstanding', new_name='Days of Sales Outstanding'),
            CuratedFeature(raw_name='netProfitMargin', new_name='Net Profit Margin'),
            CuratedFeature(raw_name='pretaxProfitMargin', new_name='Pretax Profit Margin'),
            CuratedFeature(raw_name='grossProfitMargin', new_name='Gross Profit Margin'),
            CuratedFeature(raw_name='operatingProfitMargin', new_name='Operating Profit Margin'),
            CuratedFeature(raw_name='returnOnAssets', new_name='Return on Assets'),
            CuratedFeature(raw_name='returnOnCapitalEmployed', new_name='Return on Capital Employed'),
            CuratedFeature(raw_name='returnOnEquity', new_name='Return on Equity'),
            CuratedFeature(raw_name='assetTurnover', new_name='Asset Turnover'),
            CuratedFeature(raw_name='fixedAssetTurnover', new_name='Fixed Asset Turnover'),
            CuratedFeature(raw_name='debtEquityRatio', new_name='Debt Equity Ratio'),
            CuratedFeature(raw_name='debtRatio', new_name='Debt Ratio'),
            CuratedFeature(raw_name='effectiveTaxRate', new_name='Effective Tax Rate'),
            CuratedFeature(raw_name='freeCashFlowOperatingCashFlowRatio', new_name='Free Cash Flow Operating Cash Flow Ratio'),
            CuratedFeature(raw_name='freeCashFlowPerShare', new_name='Free Cash Flow Per Share'),
            CuratedFeature(raw_name='cashPerShare', new_name='Cash Per Share'),
            CuratedFeature(raw_name='companyEquityMultiplier', new_name='Company Equity Multiplier'),
            CuratedFeature(raw_name='ebitPerRevenue', new_name='EBIT Per Revenue'),
            CuratedFeature(raw_name='enterpriseValueMultiple', new_name='Enterprise Value Multiple'),
            CuratedFeature(raw_name='operatingCashFlowPerShare', new_name='Operating Cash Flow Per Share'),
            CuratedFeature(raw_name='operatingCashFlowSalesRatio', new_name='Operating Cash Flow Sales Ratio'),
            CuratedFeature(raw_name='payablesTurnover', new_name='Payables Turnover')
            ]
