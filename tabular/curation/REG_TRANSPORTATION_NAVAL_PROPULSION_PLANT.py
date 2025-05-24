from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: naval_propulsion_plant
====
Examples: 11934
====
URL: https://www.openml.org/search?type=data&id=44969
====
Description: **Data Description**

Data have been generated from a sophisticated simulator of a Gas Turbines (GT), mounted on a Frigate characterized by a COmbined Diesel eLectric And Gas (CODLAG) propulsion plant type.

The propulsion system behaviour has been described with this parameters:
- Ship speed (linear function of the lever position lp).
- Compressor degradation coefficient kMc.
- Turbine degradation coefficient kMt.
so that each possible degradation state can be described by a combination of this triple (lp,kMt,kMc).

A series of measures (16 features) which indirectly represents the state of the system subject to performance decay has been acquired and stored in the dataset over the parameter's space.

The goal is to estimate gt_compressor_decay_state_coefficient from the given measurements.

The columns gt_compressor_inlet_air_pressure, gt_compressor_inlet_air_temperature from the original dataset were
removed because they have constant values.


**Attribute Description**

1. *lever_position*
2. *ship_speed*
3. *gas_turbine_shaft_torque*
4. *gas_turbine_rate_of_revolutions*
5. *gas_generator_rate_of_revolutions*
6. *starboard_propeller_torque*
7. *port_propeller_torque*
8. *hp_turbine_exit_temperature*
9. *gt_compressor_inlet_air_temperature*
10. *gt_compressor_outlet_air_temperature*
11. *hp_turbine_exit_pressure*
12. *gt_compressor_inlet_air_pressure*
13. *gt_compressor_outlet_air_pressure*
14. *gas_turbine_exhaust_gas_pressure*
15. *turbine_injecton_control*
16. *fuel_flow*
17. *gt_compressor_decay_state_coefficient* - target feature
18. *gt_turbine_decay_state_coefficient* - alternate target feature
====
Target Variable: gt_compressor_decay_state_coefficient (numeric, 51 distinct): ['0.95', '0.988', '0.978', '0.979', '0.98', '0.981', '0.982', '0.983', '0.984', '0.985']
====
Features:

lever_position (numeric, 9 distinct): ['1.138', '2.088', '3.144', '4.161', '5.14', '6.175', '7.148', '8.206', '9.3']
ship_speed (numeric, 9 distinct): ['3', '6', '9', '12', '15', '18', '21', '24', '27']
gas_turbine_shaft_torque (numeric, 11430 distinct): ['50992.96', '29795.666', '50993.539', '29794.302', '14724.277', '14723.563', '50994.908', '14724.099', '50994.213', '50993.745']
gas_turbine_rate_of_revolutions (numeric, 3888 distinct): ['2678.078', '2678.077', '2678.076', '2678.075', '1547.465', '1547.455', '1547.463', '1547.458', '1547.452', '1547.461']
gas_generator_rate_of_revolutions (numeric, 11834 distinct): ['6589.002', '8780.01', '8782.024', '8812.994', '8781.017', '9299.129', '9309.803', '8787.064', '9741.806', '9309.388']
starboard_propeller_torque (numeric, 4286 distinct): ['60.337', '60.334', '113.748', '60.346', '60.348', '60.331', '113.751', '60.327', '60.352', '113.786']
port_propeller_torque (numeric, 4286 distinct): ['60.337', '60.334', '113.748', '60.346', '60.348', '60.331', '113.751', '60.327', '60.352', '113.786']
hp_turbine_exit_temperature (numeric, 11772 distinct): ['574.611', '919.925', '629.503', '582.737', '586.592', '914.811', '923.29', '587.435', '581.412', '635.256']
gt_compressor_outlet_air_temperature (numeric, 11506 distinct): ['559.331', '559.238', '577.215', '563.05', '561.831', '772.08', '559.928', '779.453', '561.652', '559.283']
hp_turbine_exit_pressure (numeric, 524 distinct): ['1.391', '1.39', '1.389', '1.392', '1.388', '1.66', '1.662', '1.661', '1.659', '1.663']
gt_compressor_outlet_air_pressure (numeric, 4209 distinct): ['7.473', '7.575', '6.575', '7.443', '7.502', '7.487', '7.437', '7.479', '7.508', '7.423']
gas_turbine_exhaust_gas_pressure (numeric, 19 distinct): ['1.019', '1.02', '1.026', '1.03', '1.023', '1.036', '1.041', '1.042', '1.035', '1.022']
turbine_injecton_control (numeric, 8496 distinct): ['0.0', '12.408', '11.942', '12.008', '12.492', '12.542', '12.593', '12.61', '12.174', '12.224']
fuel_flow (numeric, 696 distinct): ['0.241', '0.242', '0.243', '0.24', '0.245', '0.244', '0.246', '0.238', '0.239', '0.247']
'''

CONTEXT = "Naval Propulsion Plant Gas Turbines"
TARGET = CuratedTarget(raw_name="gt_compressor_decay_state_coefficient",
                       new_name="Compressor Decay State Coefficient", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []