from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: FICO-HELOC-cleaned
====
Examples: 9871
====
URL: https://www.openml.org/search?type=data&id=45554
====
Description: This dataset is from the "Explainable Machine Learning Challenge":

> The Explainable Machine Learning Challenge is a collaboration between Google, FICO and academics at Berkeley, Oxford, Imperial, UC Irvine and MIT, to generate new research in the area of algorithmic explainability. Teams will be challenged to create machine learning models with both high accuracy and explainability; they will use a real-world financial dataset provided by FICO. Designers and end users of machine learning algorithms will both benefit from more interpretable and explainable algorithms. Machine learning model designers will benefit from Model explanations, written explanations describing the functioning of a trained model. These might include information about which variables or examples are particularly important, they might explain the logic used by an algorithm, and/or characterize input/output relationships between variables and predictions. We expect teams to tell the story of their model such that these explanations will be qualitatively evaluated by data scientists at FICO.

Further information can be retrieved from the [FICO website](https://community.fico.com/s/explainable-machine-learning-challenge).

**Notes**
* We have obtained the dataset from [Kaggle](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)
* This is a cleaned version of the Kaggle dataset, in which we have removed all rows that only contained `-9`, a special value according to the FAQ.
* Please request access to the data on the FICO website to obtain the full description of the features.
* In this version we have encoded the special values (-9, -8, -7) as missing values to make the data more amenable to non-tree models.
====
Target Variable: RiskPerformance (nominal, 2 distinct): ['Bad', 'Good']
====
Features:

ExternalRiskEstimate (numeric, 70 distinct): ['65.0', '66.0', '68.0', '73.0', '72.0', '70.0', '63.0', '75.0', '69.0', '80.0']
MSinceOldestTradeOpen (numeric, 763 distinct): ['178.0', '132.0', '176.0', '150.0', '165.0', '183.0', '206.0', '172.0', '169.0', '158.0']
MSinceMostRecentTradeOpen (numeric, 111 distinct): ['2', '3', '4', '5', '1', '6', '7', '8', '9', '10']
AverageMInFile (numeric, 236 distinct): ['71', '79', '74', '68', '80', '75', '84', '63', '70', '69']
NumSatisfactoryTrades (numeric, 73 distinct): ['18', '15', '16', '22', '19', '13', '14', '21', '20', '17']
NumTrades60Ever2DerogPubRec (numeric, 18 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
NumTrades90Ever2DerogPubRec (numeric, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '10']
PercentTradesNeverDelq (numeric, 71 distinct): ['100', '96', '97', '95', '94', '93', '92', '88', '90', '89']
MSinceMostRecentDelq (numeric, 4924 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '7.0', '10.0', '9.0']
MaxDelq2PublicRecLast12M (nominal, 9 distinct): ['7', '6', '4', '0', '5', '3', '1', '2', '9', '-9']
MaxDelqEver (nominal, 7 distinct): ['8', '6', '5', '2', '4', '3', '7', '-9']
NumTotalTrades (numeric, 87 distinct): ['15', '16', '17', '22', '20', '24', '21', '18', '19', '13']
NumTradesOpeninLast12M (numeric, 18 distinct): ['1', '0', '2', '3', '4', '5', '6', '7', '8', '9']
PercentInstallTrades (numeric, 95 distinct): ['33', '50', '29', '25', '38', '20', '36', '40', '27', '17']
MSinceMostRecentInqexcl7days (numeric, 2356 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
NumInqLast6M (numeric, 26 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NumInqLast6Mexcl7days (numeric, 26 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NetFractionRevolvingBurden (numeric, 312 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0', '9.0', '7.0']
NetFractionInstallBurden (numeric, 3556 distinct): ['100.0', '92.0', '83.0', '77.0', '87.0', '75.0', '95.0', '89.0', '90.0', '82.0']
NumRevolvingTradesWBalance (numeric, 185 distinct): ['2.0', '3.0', '4.0', '1.0', '5.0', '6.0', '7.0', '8.0', '0.0', '9.0']
NumInstallTradesWBalance (numeric, 879 distinct): ['2.0', '1.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
NumBank2NatlTradesWHighUtilization (numeric, 600 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
PercentTradesWBalance (numeric, 111 distinct): ['100.0', '50.0', '67.0', '75.0', '80.0', '60.0', '83.0', '71.0', '33.0', '57.0']
'''

CONTEXT = "Financial Credit Risk of Home Equity Line (HELOC)"
TARGET = CuratedTarget(raw_name="RiskPerformance", new_name="HELOC Risk Performance", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name="ExternalRiskEstimate", new_name="External Risk Estimate"),
    CuratedFeature(raw_name="MSinceOldestTradeOpen", new_name="Months Since Oldest Trade Open"),
    CuratedFeature(raw_name="MSinceMostRecentTradeOpen", new_name="Months Since Most Recent Trade Open"),
    CuratedFeature(raw_name="AverageMInFile", new_name="Average Months in File"),
    CuratedFeature(raw_name="NumSatisfactoryTrades", new_name="Number of Satisfactory Trades"),
    CuratedFeature(raw_name="NumTrades60Ever2DerogPubRec", new_name="Number of Trades 60+ Ever 2 Derogatory Public Records"),
    CuratedFeature(raw_name="NumTrades90Ever2DerogPubRec", new_name="Number of Trades 90+ Ever 2 Derogatory Public Records"),
    CuratedFeature(raw_name="PercentTradesNeverDelq", new_name="Percent of Trades Never Delinquent"),
    CuratedFeature(raw_name="MSinceMostRecentDelq", new_name="Months Since Most Recent Delinquency"),
    CuratedFeature(raw_name="MaxDelq2PublicRecLast12M", new_name="Max Delinquency in Public Records Last 12 Months"),
    CuratedFeature(raw_name="MaxDelqEver", new_name="Max Delinquency Ever"),
    CuratedFeature(raw_name="NumTotalTrades", new_name="Number of Total Trades"),
    CuratedFeature(raw_name="NumTradesOpeninLast12M", new_name="Number of Trades Open in Last 12 Months"),
    CuratedFeature(raw_name="PercentInstallTrades", new_name="Percent of Installment Trades"),
    CuratedFeature(raw_name="MSinceMostRecentInqexcl7days", new_name="Months Since Most Recent Inquiry excluding 7 Days"),
    CuratedFeature(raw_name="NumInqLast6M", new_name="Number of Inquiries Last 6 Months"),
    CuratedFeature(raw_name="NumInqLast6Mexcl7days", new_name="Number of Inquiries Last 6 Months excluding 7 Days"),
    CuratedFeature(raw_name="NetFractionRevolvingBurden", new_name="Net Fraction Revolving Burden"),
    CuratedFeature(raw_name="NetFractionInstallBurden", new_name="Net Fraction Installment Burden"),
    CuratedFeature(raw_name="NumRevolvingTradesWBalance", new_name="Number of Revolving Trades with Balance"),
    CuratedFeature(raw_name="NumInstallTradesWBalance", new_name="Number of Installment Trades with Balance"),
    CuratedFeature(raw_name="NumBank2NatlTradesWHighUtilization", new_name="Number of Bank to National Trades with High Utilization"),
    CuratedFeature(raw_name="PercentTradesWBalance", new_name="Percent of Trades with Balance"),
]
