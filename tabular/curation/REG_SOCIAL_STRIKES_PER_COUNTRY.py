from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: strikes
====
Examples: 625
====
URL: https://www.openml.org/search?type=data&id=549
====
Description: **Author**: Bruce Western (western@datacomm.iue.it)   
**Source**: [StatLib](http://lib.stat.cmu.edu/datasets/) - 1999  
**Please cite**:   

The data consist of annual observations on the level of strike volume (days lost due to industrial disputes per 1000 wage salary earners), and their covariates in 18 OECD countries from 1951-1985. The average level and variance of strike volume varies across countries. The data distribution also features a long right tail and several large outliers. 

The 7 data fields include the following variables:  
>
(1) country code;  
(2) year;  
(3) strike volume;  
(4) unemployment;  
(5) inflation;  
(6) parliamentary representation of social democratic and labor parties  
(7) a time-invariant measure of union centralization.

These data were analyzed in the forthcoming paper by Bruce Western, "Vague Theory and Model Uncertainty in Macrosociology," which is to appear in Sociological Methodology. Permission is given by the author to freely use and redistribute these data.
====
Target Variable: strike_volume (numeric, 358 distinct): ['1.0', '0.0', '2.0', '3.0', '6.0', '7.0', '8.0', '11.0', '9.0', '4.0']
====
Features:

country_code (numeric, 18 distinct): ['1', '2', '17', '16', '15', '14', '13', '12', '11', '10']
year (numeric, 35 distinct): ['1951.0', '1967.0', '1980.0', '1979.0', '1978.0', '1977.0', '1976.0', '1975.0', '1974.0', '1973.0']
unemployment (numeric, 116 distinct): ['1.2', '1.1', '0.0', '0.9', '1.7', '1.8', '1.5', '0.1', '1.6', '1.9']
inflation (numeric, 171 distinct): ['2.7', '4.3', '1.9', '3.2', '3.4', '4.2', '2.2', '1.3', '2.8', '0.0']
parliamentary_representation (numeric, 149 distinct): ['46.7', '27.0', '35.3', '42.5', '40.2', '41.0', '49.3', '52.0', '48.6', '43.8']
union_centralization (numeric, 8 distinct): ['0.375', '0.0', '0.5', '0.75', '0.25', '0.875', '1.0', '0.125']
'''

CONTEXT = "Strikes in 18 OECD countries from 1951-1985"
TARGET = CuratedTarget(raw_name="strike_volume", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []