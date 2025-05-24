from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cylinder-bands
====
Examples: 540
====
URL: https://www.openml.org/search?type=data&id=6332
====
Description: **Author**: Bob Evans, RR Donnelley & Sons Co.  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Cylinder+Bands) - August, 1995  
**Please cite**:  [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)

### Description

Cylinder bands UCI dataset - Process delays known as cylinder banding in rotogravure printing were substantially mitigated using control rules discovered by decision tree induction.
 
### Attribute Information

There are 40 attributes for 540 observations, including the class: 20 are numeric and 20 are nominal.  
There are missing values in 302 of the instances.

```
   1. timestamp: numeric;19500101 - 21001231  
   2. cylinder number: nominal  
   3. customer: nominal;  
   4. job number: nominal;   
   5. grain screened: nominal; yes, no  
   6. ink color: nominal;  key, type  
   7. proof on ctd ink:  nominal;  yes, no   
   8. blade mfg: nominal;  benton, daetwyler, uddeholm  
   9. cylinder division: nominal;  gallatin, warsaw, mattoon  
  10. paper type: nominal;  uncoated, coated, super  
  11. ink type: nominal;  uncoated, coated, cover  
  12. direct steam: nominal; use; yes, no *  
  13. solvent type: nominal;  xylol, lactol, naptha, line, other  
  14. type on cylinder:  nominal;  yes, no   
  15. press type: nominal; use; 70 wood hoe, 70 motter, 70 albert, 94 motter  
  16. press: nominal;  821, 802, 813, 824, 815, 816, 827, 828  
  17. unit number: nominal;  1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
  18. cylinder size: nominal;  catalog, spiegel, tabloid  
  19. paper mill location: nominal; north us, south us, canadian, 
      scandanavian, mid european  
  20. plating tank: nominal; 1910, 1911, other  
  21. proof cut: numeric;  0-100  
  22. viscosity: numeric;  0-100  
  23. caliper: numeric;  0-1.0  
  24. ink temperature: numeric;  5-30  
  25. humifity: numeric;  5-120  
  26. roughness: numeric;  0-2  
  27. blade pressure: numeric;  10-75  
  28. varnish pct: numeric;  0-100  
  29. press speed: numeric;  0-4000  
  30. ink pct: numeric;  0-100  
  31. solvent pct: numeric;  0-100  
  32. ESA Voltage: numeric;  0-16  
  33. ESA Amperage: numeric;  0-10  
  34. wax: numeric ;  0-4.0  
  35. hardener:  numeric; 0-3.0  
  36. roller durometer:  numeric;  15-120  
  37. current density:  numeric;  20-50  
  38. anode space ratio:  numeric;  70-130  
  39. chrome content: numeric; 80-120  
  40. band type: nominal; class; band, no band  
```

**Notes**:  
* cylinder number is an identifier and should be ignored when modeling the data
* data set consists of 540 observations. UCI explanation states 541, which is wrong. 

### Relevant Papers

Evans, B., and Fisher, D. (1994). Overcoming process delays with decision tree induction. IEEE Expert, Vol. 9, No. 1, 60--66.
====
Target Variable: band_type (nominal, 2 distinct): ['noband', 'band']
====
Features:

customer (nominal, 71 distinct): ['kmart', 'modmat', 'target', 'tvguide', 'wards', 'toysrus', 'ames', 'roses', 'bestprod', 'eckerd']
job_number (numeric, 262 distinct): ['47103', '47105', '47203', '47202', '36197', '37386', '47104', '37365', '34493', '37352']
grain_screened (nominal, 3 distinct): ['no', 'yes', '45']
ink_color (nominal, 1 distinct): ['key', '0.200']
proof_on_ctd_ink (nominal, 3 distinct): ['yes', 'no', '17']
blade_mfg (nominal, 3 distinct): ['benton', 'uddeholm', '84']
cylinder_division (nominal, 1 distinct): ['gallatin', '0.8125']
paper_type (nominal, 3 distinct): ['uncoated', 'coated', 'super', '27']
ink_type (nominal, 3 distinct): ['coated', 'uncoated', 'cover']
direct_steam (nominal, 3 distinct): ['no', 'yes', '1865']
solvent_type (nominal, 4 distinct): ['line', 'xylol', 'naptha']
type_on_cylinder (nominal, 3 distinct): ['yes', 'no']
press_type (nominal, 4 distinct): ['motter94', 'woodhoe70', 'albert70', 'motter70']
press (nominal, 8 distinct): ['816', '815', '821', '824', '802', '827', '813', '828']
unit_number (numeric, 7 distinct): ['2', '9', '7', '1', '5', '10', '6']
cylinder_size (nominal, 4 distinct): ['tabloid', 'catalog', 'spiegel', '0.7']
paper_mill_location (nominal, 6 distinct): ['northus', 'canadian', 'mideuropean', 'scandanavian', 'southus']
plating_tank (nominal, 3 distinct): ['1910', '1911', '40']
proof_cut (numeric, 28 distinct): ['40.0', '50.0', '45.0', '35.0', '30.0', '55.0', '47.5', '60.0', '37.5', '65.0']
viscosity (numeric, 38 distinct): ['54.0', '43.0', '42.0', '56.0', '50.0', '60.0', '45.0', '47.0', '58.0', '41.0']
caliper (nominal, 21 distinct): ['0.2', '0.3', '0.233', '0.267', '0.333', '0.367', '0.4', '0.300', '0.433', '0.200']
ink_temperature (numeric, 66 distinct): ['15.0', '16.0', '14.0', '15.5', '17.0', '14.5', '16.5', '16.1', '13.0', '14.9']
humifity (numeric, 43 distinct): ['80.0', '70.0', '78.0', '75.0', '76.0', '72.0', '82.0', '74.0', '84.0', '85.0']
roughness (numeric, 19 distinct): ['0.75', '0.625', '1.0', '0.5', '0.8125', '0.25', '0.875', '0.5625', '0.375', '0.3125']
blade_pressure (numeric, 37 distinct): ['30.0', '20.0', '25.0', '28.0', '50.0', '32.0', '35.0', '34.0', '40.0', '24.0']
varnish_pct (numeric, 123 distinct): ['0.0', '5.6', '10.5', '3.4', '15.0', '11.1', '8.0', '15.8', '7.6', '14.1']
press_speed (numeric, 84 distinct): ['1800.0', '2000.0', '1600.0', '1700.0', '1900.0', '1500.0', '1750.0', '2400.0', '2100.0', '1650.0']
ink_pct (numeric, 82 distinct): ['58.8', '62.5', '55.6', '52.6', '56.8', '54.3', '50.0', '53.2', '58.1', '47.6']
solvent_pct (numeric, 116 distinct): ['41.2', '37.5', '38.9', '39.8', '40.0', '33.3', '38.1', '35.0', '36.9', '38.5']
ESA_Voltage (numeric, 18 distinct): ['0.0', '2.0', '4.0', '1.0', '3.0', '5.0', '8.0', '10.0', '6.0', '12.0']
ESA_Amperage (numeric, 5 distinct): ['0.0', '4.0', '0.5', '6.0']
wax (numeric, 31 distinct): ['2.5', '3.0', '2.0', '1.5', '2.7', '2.4', '2.8', '2.3', '2.6', '2.9']
hardener (numeric, 30 distinct): ['1.0', '0.8', '0.9', '0.7', '0.6', '1.5', '1.1', '0.5', '1.3', '1.2']
roller_durometer (numeric, 13 distinct): ['30.0', '40.0', '33.0', '35.0', '34.0', '38.0', '28.0', '32.0', '45.0', '50.0']
current_density (nominal, 8 distinct): ['40', '35', '33', '30', '45', '37', '42']
anode_space_ratio (numeric, 81 distinct): ['100.0', '106.45', '103.125', '103.22', '106.25', '103.3', '110.0', '107.4', '106.66', '93.75']
chrome_content (nominal, 4 distinct): ['100', '90', '95']
'''

CONTEXT = "Process Delays - Cylinder Banding in Rotogravure Printing"
TARGET = CuratedTarget(raw_name="band_type", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["job_number"]
FEATURES = []