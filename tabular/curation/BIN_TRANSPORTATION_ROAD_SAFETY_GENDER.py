from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: road-safety
====
Examples: 111762
====
URL: https://www.openml.org/search?type=data&id=45038
====
Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark, transformed in the same way. This dataset belongs to the "classification on both numerical and categorical features" benchmark. 
 
  Original link: https://openml.org/d/42803 
 
 Original description: 
 
Data reported to the police about the circumstances of personal injury road accidents in Great Britain from 1979, and the maker and model information of vehicles involved in the respective accident.

This version includes data up to 2015.
====
Target Variable: SexofDriver (nominal, 2 distinct): ['0', '1']
====
Features:

Vehicle_Reference_df_res (numeric, 32 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '10', '9']
Vehicle_Type (numeric, 15 distinct): ['9', '19', '3', '5', '8', '11', '21', '2', '4', '20']
Vehicle_Manoeuvre (numeric, 18 distinct): ['18', '9', '4', '3', '17', '5', '16', '7', '2', '10']
Vehicle_Location-Restricted_Lane (numeric, 10 distinct): ['0', '9', '2', '6', '8', '7', '4', '1', '5', '3']
Hit_Object_in_Carriageway (numeric, 12 distinct): ['0', '10', '4', '7', '11', '9', '1', '12', '6', '8']
Hit_Object_off_Carriageway (numeric, 12 distinct): ['0', '10', '4', '11', '9', '6', '1', '7', '2', '3']
Was_Vehicle_Left_Hand_Drive? (nominal, 2 distinct): ['0', '1']
Age_of_Driver (numeric, 84 distinct): ['25', '24', '30', '23', '28', '27', '26', '21', '22', '35']
Age_Band_of_Driver (numeric, 9 distinct): ['6', '7', '8', '5', '9', '4', '10', '11', '3']
Engine_Capacity_(CC) (numeric, 873 distinct): ['1598.0', '1242.0', '998.0', '1560.0', '1995.0', '1968.0', '1997.0', '1896.0', '1596.0', '1796.0']
Propulsion_Code (numeric, 8 distinct): ['1', '2', '8', '7', '12', '6', '5', '10']
Age_of_Vehicle (numeric, 56 distinct): ['1', '8', '2', '9', '11', '10', '12', '7', '3', '5']
Location_Easting_OSGR (numeric, 44941 distinct): ['455113.0', '478489.0', '453898.0', '533650.0', '412672.0', '517570.0', '428502.0', '518040.0', '567845.0', '548750.0']
Location_Northing_OSGR (numeric, 46504 distinct): ['220016.0', '428516.0', '177930.0', '192397.0', '286682.0', '183820.0', '175310.0', '182330.0', '181760.0', '398420.0']
Longitude (numeric, 60608 distinct): ['-1.2008', '-0.8113', '-1.2227', '-1.8149', '0.1396', '0.4212', '-0.4952', '-0.2994', '-2.8577', '-0.2656']
Latitude (numeric, 60310 distinct): ['51.876', '53.7471', '51.6278', '52.4779', '53.8776', '51.5843', '52.1916', '53.3478', '52.5973', '51.4572']
Police_Force (numeric, 44 distinct): ['1', '20', '43', '13', '50', '4', '14', '6', '42', '5']
Number_of_Vehicles (numeric, 14 distinct): ['2', '3', '1', '4', '5', '6', '37', '7', '8', '10']
Number_of_Casualties (numeric, 18 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '36', '9']
Local_Authority_(District) (numeric, 324 distinct): ['300.0', '204.0', '635.0', '596.0', '91.0', '215.0', '200.0', '211.0', '169.0', '231.0']
1st_Road_Number (numeric, 3045 distinct): ['0.0', '1.0', '6.0', '4.0', '40.0', '38.0', '5.0', '62.0', '41.0', '61.0']
2nd_Road_Number (numeric, 2920 distinct): ['0.0', '6.0', '1.0', '57.0', '7302.0', '38.0', '5.0', '4.0', '7201.0', '40.0']
Urban_or_Rural_Area (nominal, 2 distinct): ['0', '1']
Vehicle_Reference_df (numeric, 23 distinct): ['1', '2', '3', '4', '5', '6', '7', '10', '27', '22']
Casualty_Reference (numeric, 35 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
Sex_of_Casualty (nominal, 2 distinct): ['0', '1']
Age_of_Casualty (numeric, 100 distinct): ['23', '22', '21', '25', '19', '24', '20', '18', '26', '27']
Age_Band_of_Casualty (numeric, 11 distinct): ['6', '7', '8', '5', '4', '9', '10', '11', '3', '2']
Pedestrian_Location (numeric, 11 distinct): ['0', '5', '1', '9', '4', '6', '10', '8', '7', '2']
Pedestrian_Movement (numeric, 10 distinct): ['0', '1', '3', '9', '2', '4', '5', '8', '7', '6']
Casualty_Type (numeric, 21 distinct): ['9', '0', '3', '1', '5', '19', '11', '8', '2', '4']
Casualty_IMD_Decile (numeric, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
'''

CONTEXT = "Road Safety by Driver Gender"
TARGET = CuratedTarget(raw_name="SexofDriver", new_name="Driver Gender", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []