from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: markusschmitz/museums/museums_prep.csv
====
Examples: 33072
====
URL: https://www.kaggle.com/markusschmitz/museums/museums_prep.csv
====
Description: 
Museums
(https://www.kaggle.com/datasets/markusschmitz/museums)
General information on the US museums. The task is to predict the revenues across the museums.

====
Features:

Unnamed: 0 (int64, 33072 distinct): ['0', '22027', '22057', '22056', '22055', '22054', '22053', '22052', '22051', '22050']
Museum ID (int64, 33072 distinct): ['8400200098', '8403600057', '8403602087', '8403600031', '8403600704', '8403601123', '8409500212', '8403602348', '8403600847', '8403600602']
Museum Name (object, 31546 distinct): ['PLANETARIUM', 'ART GALLERY', 'STARLAB PLANETARIUM', 'WASHINGTON COUNTY HISTORICAL SOCIETY', 'MUSEUM OF ANTHROPOLOGY', 'UNIVERSITY ART GALLERY', 'UNION COUNTY HISTORICAL SOCIETY', 'JACKSON COUNTY HISTORICAL SOCIETY', 'FINE ARTS GALLERY', 'CARROLL COUNTY HISTORICAL SOCIETY']
Legal Name (object, 30430 distinct): ['PLANETARIUM', 'ART GALLERY', 'STARLAB PLANETARIUM', 'PRESIDENT AND FELLOWS OF HARVARD COLLEGE', 'REGENTS OF THE UNIVERSITY OF MICHIGAN', 'REGENTS OF THE UNIVERSITY OF CALIFORNIA', 'UNIVERSITY OF MAINE SYSTEM INC', 'MICHIGAN STATE UNIVERSITY', 'JACKSON COUNTY HISTORICAL SOCIETY', 'REGENTS OF THE UNIVERSITY OF CALIFORNIA AT BERKELEY']
Alternate Name (object, 1883 distinct): ['REGENTS OF THE UNIVERSITY OF CALIFORNIA  LOS ANGELES', 'SMITHSONIAN INSTITUTION', 'GALLERY OF CONTEMPORARY ART', 'UNIVERSITY OF CALIFORNIA  DAVIS', 'AFRICAN AMERICAN HERITAGE FOUNDATION', 'UNIVERSITY OF ILLINOIS', 'RUTGERS  THE STATE UNIVERSITY OF NEW JERSEY', 'MUSEUM OF THE CITY OF NEW YORK', 'CENTER FOR ARTS AND SCIENCES OF WEST VIRGINIA', 'PRESIDENT BENJAMIN HARRISON FOUNDATION']
Museum Type (object, 9 distinct): ['HISTORIC PRESERVATION', 'GENERAL MUSEUM', 'ART MUSEUM', 'HISTORY MUSEUM', 'ARBORETUM, BOTANICAL GARDEN, OR NATURE CENTER', 'SCIENCE & TECHNOLOGY MUSEUM OR PLANETARIUM', 'ZOO, AQUARIUM, OR WILDLIFE CONSERVATION', "CHILDREN'S MUSEUM", 'NATURAL HISTORY MUSEUM']
Institution Name (object, 1581 distinct): ['PENNSYLVANIA STATE UNIVERSITY', 'HARVARD UNIVERSITY', 'ARIZONA STATE UNIVERSITY', 'UNIVERSITY OF MICHIGAN', 'UNIVERSITY OF ARIZONA', 'MICHIGAN STATE UNIVERSITY', 'UNIVERSITY OF WISCONSIN-MADISON', 'UNIVERSITY OF VIRGINIA', 'UNIVERSITY OF NEBRASKA-LINCOLN', 'UNIVERSITY OF CALIFORNIA, BERKELEY']
Street Address (Administrative Location) (object, 25493 distinct): ['603 W JACKSON', 'PO BOX 1', 'PO BOX 25', 'PO BOX 3', 'PO BOX 127', 'PO BOX 12', 'PO BOX 44', 'PO BOX 2', 'PO BOX 125', 'PO BOX 5']
City (Administrative Location) (object, 8621 distinct): ['NEW YORK', 'WASHINGTON', 'CHICAGO', 'PHILADELPHIA', 'LOS ANGELES', 'PORTLAND', 'HOUSTON', 'BALTIMORE', 'SAN FRANCISCO', 'SPRINGFIELD']
State (Administrative Location) (object, 51 distinct): ['CA', 'NY', 'TX', 'PA', 'OH', 'IL', 'FL', 'MI', 'MA', 'VA']
Zip Code (Administrative Location) (object, 15522 distinct): ['74743', '92101', '19106', '10011', '2840', '17325', '2138', '33040', '16802', '70130']
Street Address (Physical Location) (object, 8769 distinct): ['603 W JACKSON', 'PO BOX 33', 'PO BOX 351', 'PO BOX 34', 'MAIN STREET', 'PO BOX 321', 'PO BOX 356', 'PO BOX 333', 'PO BOX 334', 'PO BOX 345']
City (Physical Location) (object, 4480 distinct): ['NEW YORK', 'WASHINGTON', 'SPRINGFIELD', 'LOS ANGELES', 'CHICAGO', 'PHILADELPHIA', 'PORTLAND', 'SAN FRANCISCO', 'BALTIMORE', 'RICHMOND']
State (Physical Location) (object, 51 distinct): ['NY', 'CA', 'TX', 'PA', 'OH', 'IL', 'FL', 'MA', 'MI', 'VA']
Zip Code (Physical Location) (float64, 7011 distinct): ['74743.0', '92101.0', '77550.0', '87504.0', '1103.0', '17603.0', '37203.0', '24450.0', '21202.0', '72712.0']
Phone Number (object, 20517 distinct): ['6605621212', '2026331000', '6174959400', '8175157607', '2547101110', '6079375281', '7342407780', '3192346357', '8157773310', '7065078800']
Latitude (float64, 27276 distinct): ['40.7979', '42.3698', '42.2529', '44.8022', '42.7356', '33.6912', '38.0359', '37.8743', '41.4225', '39.5511']
Longitude (float64, 27463 distinct): ['-77.8627', '-71.1122', '-83.7389', '-68.7715', '-84.4935', '-112.5385', '-78.5049', '-122.2664', '-105.6072', '-83.0322']
Locale Code (NCES) (float64, 4 distinct): ['4.0', '1.0', '2.0', '3.0']
County Code (FIPS) (float64, 300 distinct): ['1.0', '3.0', '37.0', '31.0', '13.0', '5.0', '61.0', '17.0', '9.0', '19.0']
State Code (FIPS) (float64, 54 distinct): ['6.0', '36.0', '48.0', '42.0', '39.0', '17.0', '12.0', '26.0', '25.0', '51.0']
Region Code (AAM) (int64, 6 distinct): ['4', '3', '2', '6', '5', '1']
Employer ID Number (object, 24921 distinct): ['526002033', '396006492', '736017987', '860196696', '246000376', '146013200', '470491233', '42103580', '386006309', '946001347']
Tax Period (float64, 67 distinct): ['201312.0', '201412.0', '201406.0', '201306.0', '201409.0', '201405.0', '201212.0', '201403.0', '201404.0', '201309.0']
Income (float64, 10775 distinct): ['0.0', '1.0', '83181439574.0', '844590000.0', '141683600.0', '867057457.0', '415452342.0', '7943643646.0', '1014902481.0', '1187073882.0']
Revenue (float64, 10167 distinct): ['0.0', '5840349457.0', '130058221.0', '139978.0', '840046127.0', '3688471185.0', '379772463.0', '543405959.0', '1371587031.0', '938599729.0']
'''

CONTEXT = "General information on the US museums"
TARGET = CuratedTarget(raw_name="Revenue", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Unnamed: 0", "Museum ID", "Employer ID Number", "Income"]
FEATURES = []

DESCRIPTION = '''
Museums
(https://www.kaggle.com/datasets/markusschmitz/museums)
General information on the US museums. The task is to predict the revenues across the museums.
'''