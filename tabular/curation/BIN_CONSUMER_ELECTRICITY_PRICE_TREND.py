from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: electricity
====
Examples: 38474
====
URL: https://www.openml.org/search?type=data&id=44156
====
Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark,  
                          transformed in the same way. This dataset belongs to the "classification on categorical and
                          numerical features" benchmark. Original description: 
 
**Author**: M. Harries, J. Gama, A. Bifet  
**Source**: [Joao Gama](http://www.inescporto.pt/~jgama/ales/ales_5.html) - 2009  
**Please cite**: None  

**Electricity** is a widely used dataset described by M. Harries and analyzed by J. Gama (see papers below). This data was collected from the Australian New South Wales Electricity Market. In this market, prices are not fixed and are affected by demand and supply of the market. They are set every five minutes. Electricity transfers to/from the neighboring state of Victoria were done to alleviate fluctuations.

The dataset (originally named ELEC2) contains 45,312 instances dated from 7 May 1996 to 5 December 1998. Each example of the dataset refers to a period of 30 minutes, i.e. there are 48 instances for each time period of one day. Each example on the dataset has 5 fields, the day of week, the time stamp, the New South Wales electricity demand, the Victoria electricity demand, the scheduled electricity transfer between states and the class label. The class label identifies the change of the price (UP or DOWN) in New South Wales relative to a moving average of the last 24 hours (and removes the impact of longer term price trends). 

The data was normalized by A. Bifet.

### Attribute information  
* Date: date between 7 May 1996 to 5 December 1998. Here normalized between 0 and 1
* Day: day of the week (1-7)
* Period: time of the measurement (1-48) in half hour intervals over 24 hours. Here normalized between 0 and 1
* NSWprice: New South Wales electricity price, normalized between 0 and 1
* NSWdemand: New South Wales electricity demand, normalized between 0 and 1
* VICprice: Victoria electricity price, normalized between 0 and 1
* VICdemand: Victoria electricity demand, normalized between 0 and 1
* transfer: scheduled electricity transfer between both states, normalized between 0 and 1

### Relevant papers  
M. Harries. Splice-2 comparative evaluation: Electricity pricing. Technical report, The University of South Wales, 1999.  
J. Gama, P. Medas, G. Castillo, and P. Rodrigues. Learning with drift detection. In SBIA Brazilian Symposium on Artificial Intelligence, pages 286-295, 2004.
====
Target Variable: class (nominal, 2 distinct): ['DOWN', 'UP']
====
Features:

date (numeric, 933 distinct): ['0.4423', '0.8981', '0.8762', '0.4736', '0.8937', '0.8717', '0.8673', '0.8804', '0.8893', '0.8806']
day (nominal, 7 distinct): ['1', '0', '2', '4', '3', '5', '6']
period (numeric, 48 distinct): ['0.383', '0.4043', '0.4468', '0.4255', '0.766', '0.7447', '0.7872', '0.3191', '0.4681', '0.4894']
nswprice (numeric, 4041 distinct): ['0.0748', '0.0552', '0.0584', '0.0289', '0.0417', '0.0414', '0.0294', '0.0288', '0.0569', '0.0416']
nswdemand (numeric, 5191 distinct): ['0.4786', '0.4871', '0.4896', '0.4881', '0.4805', '0.4511', '0.4732', '0.5104', '0.4897', '0.4759']
vicprice (numeric, 3764 distinct): ['0.0035', '0.0008', '0.002', '0.001', '0.0008', '0.0008', '0.002', '0.0021', '0.002', '0.0019']
vicdemand (numeric, 2799 distinct): ['0.4229', '0.5621', '0.4946', '0.551', '0.5161', '0.5067', '0.5085', '0.34', '0.4381', '0.4676']
transfer (numeric, 1858 distinct): ['0.4149', '0.5005', '0.7658', '0.8382', '0.6342', '0.6781', '0.857', '0.8417', '0.7219', '0.6627']
'''

CONTEXT = "Electricity Price Trend"
TARGET = CuratedTarget(raw_name="class", new_name="Price Trend", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="nswprice", new_name="NSW Price"),
            CuratedFeature(raw_name="nswdemand", new_name="NSW Demand"),
            CuratedFeature(raw_name="vicprice", new_name="VIC Price"),
            CuratedFeature(raw_name="vicdemand", new_name="VIC Demand"),
            CuratedFeature(raw_name="transfer", new_name="Scheduler Transfer between States")
            ]