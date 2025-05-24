from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: energy_efficiency
====
Examples: 768
====
URL: https://www.openml.org/search?type=data&id=44960
====
Description: **Data Description**

This dataset looked into assessing the heating load and cooling load requirements of buildings (that is, energy efficiency) as a function of building parameters.
Energy analysis is performed using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. Various settings as functions of the afore-mentioned characteristics are simulated to obtain 768 building shapes (number of observations in the dataset).

**Attribute Description**

All features describe different properties for the building.

1. *relative_compactness*
2. *surface_area*
3. *wall_area*
4. *roof_area*
5. *overall_height*
6. *orientation*
7. *glazing_area*
8. *glazing_area_distribution*
9. *heating_load* - one possible option for target feature
10. *cooling_load* - one possible option for target feature
====
Target Variable: heating_load (numeric, 587 distinct): ['15.16', '13.0', '15.23', '28.15', '14.6', '32.31', '10.68', '15.55', '15.09', '12.93']
====
Features:

relative_compactness (numeric, 12 distinct): ['0.98', '0.9', '0.86', '0.82', '0.79', '0.76', '0.74', '0.71', '0.69', '0.66']
surface_area (numeric, 12 distinct): ['514.5', '563.5', '588.0', '612.5', '637.0', '661.5', '686.0', '710.5', '735.0', '759.5']
wall_area (numeric, 7 distinct): ['294.0', '318.5', '343.0', '416.5', '245.0', '269.5', '367.5']
roof_area (numeric, 4 distinct): ['220.5', '147.0', '122.5', '110.25']
overall_height (numeric, 2 distinct): ['7.0', '3.5']
orientation (numeric, 4 distinct): ['2', '3', '4', '5']
glazing_area (numeric, 4 distinct): ['0.1', '0.25', '0.4', '0.0']
glazing_area_distribution (numeric, 6 distinct): ['1', '2', '3', '4', '5', '0']
'''

CONTEXT = "Energy Efficiency of Buildings"
TARGET = CuratedTarget(raw_name="heating_load", new_name="Heating Load", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []