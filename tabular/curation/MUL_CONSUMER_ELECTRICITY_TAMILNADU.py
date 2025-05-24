from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: tamilnadu-electricity
====
Examples: 45781
====
URL: https://www.openml.org/search?type=data&id=40985
====
Description: **Author**: K.Kalyani.    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Tamilnadu+Electricity+Board+Hourly+Readings) - 2013  
**Please cite**:   

__Major changes w.r.t. version 2: ignored variable 3 in this upload as this seems to be ea perfect predictor.__

Tamilnadu Electricity Board Hourly Readings dataset. 

Real-time readings were collected from residential, commercial, industrial and agriculture to find the accuracy consumption in Tamil Nadu, around Thanajvur. 

**Note**: the attribute Sector was removed from original source since it was constant to all instances.
**Note**: the attribute serviceID should be removed when predicting the target from W and VA.

### Attribute Information:
1 - ForkVA (V1) : Voltage-Ampere readings
2 - ForkW (V2) : Wattage readings
4 - Type (Class): 
- Bank  
- AutomobileIndustry 
- BpoIndustry   
- CementIndustry   
- Farmers1   
- Farmers2   
- HealthCareResources 
- TextileIndustry 
- PoultryIndustry 
- Residential(individual)  
- Residential(Apartments)    
- FoodIndustry   
- ChemicalIndustry   
- Handlooms   
- FertilizerIndustry   
- Hostel   
- Hospital   
- Supermarket   
- Theatre   
- University
====
Target Variable: Class (nominal, 20 distinct): ['9', '4', '13', '20', '6', '3', '7', '10', '11', '2']
====
Features:

V1 (numeric, 44778 distinct): ['0.0125', '0.6692', '0.0747', '0.2916', '0.4079', '0.8059', '0.8836', '0.1372', '0.0138', '0.2791']
V2 (numeric, 44777 distinct): ['0.9778', '0.3818', '0.8947', '0.3413', '0.3177', '0.0527', '0.4791', '0.4825', '0.4581', '0.9168']
'''

CONTEXT = "Electricity consumption in Tamil Nadu"
TARGET = CuratedTarget(raw_name="Class", new_name="Sector", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={'1': 'Bank',
                                      '2': 'Automobile Industry',
                                      '3': 'Bpo Industry',
                                      '4': 'Cement Industry',
                                      '5': 'Farmers1',
                                      '6': 'Farmers2',
                                      '7': 'HealthCare Resources',
                                      '8': 'Textile Industry',
                                      '9': 'PoultryIndustry',
                                      '10': 'Residential individual',
                                      '11': 'Residential Apartments',
                                      '12': 'Food Industry',
                                      '13': 'Chemical Industry',
                                      '14': 'Handlooms',
                                      '15': 'Fertilizer Industry',
                                      '16': 'Hostel',
                                      '17': 'Hospital',
                                      '18': 'Supermarket',
                                      '19': 'Theatre',
                                      '20': 'University'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="ForkVA - Voltage-Ampere readings"),
            CuratedFeature(raw_name="V2", new_name="ForkW - Wattage readings")]