from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cars
====
Examples: 804
====
URL: https://www.openml.org/search?type=data&id=44994
====
Description: **Data Description**

Data frame of the suggested retail prices (column *Price*) and various characteristics of each car.

For this data set, a representative sample of over eight hundred 2005 GM cars were selected, then retail prices were calculated from the tables provided in the 2005 Central Edition of the Kelly Blue Book.

**Attribute Description**

All features describe different self-explanatory characteristics for the cars.

1. *Price* - target feature
2. *Mileage*
3. *Cylinder*
4. *Doors*
5. *Cruise*
6. *Sound*
7. *Leather*
8. *Buick*
9. *Cadillac*
10. *Chevy*
11. *Pontiac*
12. *Saab*
13. *Saturn*
14. *convertible*
15. *coupe*
16. *hatchback*
17. *sedan*
18. *wagon*
====
Target Variable: Price (numeric, 798 distinct): ['16507.07', '15979.01', '16256.24', '10921.95', '11539.05', '14077.97', '17418.07', '23159.54', '26302.07', '23348.02']
====
Features:

Mileage (numeric, 791 distinct): ['18910.0', '22964.0', '10014.0', '24568.0', '9795.0', '22740.0', '10555.0', '26034.0', '26328.0', '18661.0']
Cylinder (numeric, 3 distinct): ['4', '6', '8']
Doors (numeric, 2 distinct): ['4', '2']
Cruise (numeric, 2 distinct): ['1', '0']
Sound (numeric, 2 distinct): ['1', '0']
Leather (numeric, 2 distinct): ['1', '0']
Buick (numeric, 2 distinct): ['0', '1']
Cadillac (numeric, 2 distinct): ['0', '1']
Chevy (numeric, 2 distinct): ['0', '1']
Pontiac (numeric, 2 distinct): ['0', '1']
Saab (numeric, 2 distinct): ['0', '1']
Saturn (numeric, 2 distinct): ['0', '1']
convertible (numeric, 2 distinct): ['0', '1']
coupe (numeric, 2 distinct): ['0', '1']
hatchback (numeric, 2 distinct): ['0', '1']
sedan (numeric, 2 distinct): ['1', '0']
wagon (numeric, 2 distinct): ['0', '1']
'''

CONTEXT = "Car GM Price"
TARGET = CuratedTarget(raw_name="Price", new_name="Retail Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []