from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: blood-transfusion-service-center
====
Examples: 748
====
URL: https://www.openml.org/search?type=data&id=1464
====
Description: **Author**: Prof. I-Cheng Yeh  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)  
**Please cite**: Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence", Expert Systems with Applications, 2008.   

**Blood Transfusion Service Center Data Set**  
Data taken from the Blood Transfusion Service Center in Hsin-Chu City in Taiwan -- this is a classification problem.

To demonstrate the RFMTC marketing model (a modified version of RFM), this study adopted the donor database of Blood Transfusion Service Center in Hsin-Chu City in Taiwan. The center passes their blood transfusion service bus to one university in Hsin-Chu City to gather blood donated about every three months. To build an FRMTC model, we selected 748 donors at random from the donor database. 

### Attribute Information  
* V1: Recency - months since last donation
* V2: Frequency - total number of donation
* V3: Monetary - total blood donated in c.c.
* V4: Time - months since first donation), and a binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).

The target attribute is a binary variable representing whether he/she donated blood in March 2007 (2 stands for donating blood; 1 stands for not donating blood).
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 31 distinct): ['2', '4', '11', '14', '16', '23', '21', '9', '3', '1']
V2 (numeric, 33 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
V3 (numeric, 33 distinct): ['250.0', '500.0', '750.0', '1000.0', '1250.0', '1500.0', '1750.0', '2000.0', '2250.0', '2750.0']
V4 (numeric, 78 distinct): ['4', '16', '14', '23', '2', '28', '26', '11', '35', '21']
'''

CONTEXT = "Blood Transfusion Service Center in Taiwan"
TARGET = CuratedTarget(raw_name="Class", new_name="Blood Donation", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "No", "2": "Yes"})
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name="V1", new_name="Recency (months since last donation)"),
            CuratedFeature(raw_name="V2", new_name="Frequency (total number of donation)"),
            CuratedFeature(raw_name="V3", new_name="Monetary (total blood donated in c.c.)"),
            CuratedFeature(raw_name="V4", new_name="Time (months since first donation)"), ]