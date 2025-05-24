from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: climate-model-simulation-crashes
====
Examples: 540
====
URL: https://www.openml.org/search?type=data&id=40994
====
Description: **Author**: D. Lucas, R. Klein, J. Tannahill, D. Ivanova, S. Brandon, D. Domyancic, Y. Zhang.  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes)  
**Please Cite**: Lucas, D. D., Klein, R., Tannahill, J., Ivanova, D., Brandon, S., Domyancic, D., and Zhang, Y.: Failure analysis of parameter-induced simulation crashes in climate models, Geosci. Model Dev. Discuss., 6, 585-623, [Web Link](http://www.geosci-model-dev-discuss.net/6/585/2013/gmdd-6-585-2013.html), 2013.  

__Major changes w.r.t. version 1: deactivated first two variables as they describe the batch of the experiments and should not be used for prediction. Also transformed the target from numeric to factor type.__


### Source

D. Lucas (ddlucas .at. alum.mit.edu), Lawrence Livermore National Laboratory; R. Klein (rklein .at. astron.berkeley.edu), Lawrence Livermore National Laboratory & U.C. Berkeley; J. Tannahill (tannahill1 .at. llnl.gov), Lawrence Livermore National Laboratory; D. Ivanova (ivanova2 .at. llnl.gov), Lawrence Livermore National Laboratory; S. Brandon (brandon1 .at. llnl.gov), Lawrence Livermore National Laboratory; D. Domyancic (domyancic1 .at. llnl.gov), Lawrence Livermore National Laboratory; Y. Zhang (zhang24 .at. llnl.gov), Lawrence Livermore National Laboratory .

This data was constructed using LLNL's UQ Pipeline, was created under the auspices of the US Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344, was funded by LLNL's Uncertainty Quantification Strategic Initiative Laboratory Directed Research and Development Project under tracking code 10-SI-013, and is released under UCRL number LLNL-MISC-633994.


### Data Set Information

This dataset contains records of simulation crashes encountered during climate model uncertainty quantification (UQ) ensembles. Ensemble members were constructed using a Latin hypercube method in LLNL's UQ Pipeline software system to sample the uncertainties of 18 model parameters within the Parallel Ocean Program (POP2) component of the Community Climate System Model (CCSM4). Three separate Latin hypercube ensembles were conducted, each containing 180 ensemble members. 46 out of the 540 simulations failed for numerical reasons at combinations of parameter values. The goal is to use classification to predict simulation outcomes (fail or succeed) from input parameter values, and to use sensitivity analysis and feature selection to determine the causes of simulation crashes. Further details about the data and methods are given in the publication 'Failure Analysis of Parameter-Induced Simulation Crashes in Climate Models,' Geoscientific Model Development [(Web Link)](doi:10.5194/gmdd-6-585-2013).


### Attribute Information

The goal is to predict climate model simulation outcomes (column 19, fail or succeed) given scaled values of climate model input parameters (columns 1-18). 

- Columns 3-20: values of 18 climate model parameters scaled in the interval [0, 1] 

- Column 21: simulation outcome (0 = failure, 1 = success)

Relevant Papers:

Lucas, D. D., Klein, R., Tannahill, J., Ivanova, D., Brandon, S., Domyancic, D., and Zhang, Y.: Failure analysis of parameter-induced simulation crashes in climate models, Geosci. Model Dev. Discuss., 6, 585-623, [Web Link](http://www.geosci-model-dev-discuss.net/6/585/2013/gmdd-6-585-2013.html), 2013.
====
Target Variable: outcome (nominal, 2 distinct): ['1', '0']
====
Features:

vconst_corr (numeric, 540 distinct): ['0.859', '0.3693', '0.3923', '0.5917', '0.0491', '0.7724', '0.6416', '0.7496', '0.9676', '0.7143']
vconst_2 (numeric, 540 distinct): ['0.9278', '0.281', '0.5487', '0.0531', '0.1386', '0.0355', '0.7212', '0.1781', '0.5419', '0.7924']
vconst_3 (numeric, 540 distinct): ['0.2529', '0.5153', '0.8436', '0.067', '0.9751', '0.1018', '0.9329', '0.428', '0.1331', '0.79']
vconst_4 (numeric, 540 distinct): ['0.2988', '0.5333', '0.5427', '0.2098', '0.7832', '0.5187', '0.4382', '0.6315', '0.8516', '0.9067']
vconst_5 (numeric, 540 distinct): ['0.1705', '0.7144', '0.6057', '0.5956', '0.4946', '0.5133', '0.3759', '0.018', '0.629', '0.863']
vconst_7 (numeric, 540 distinct): ['0.7359', '0.5052', '0.9044', '0.8874', '0.9445', '0.4242', '0.3076', '0.8596', '0.9218', '0.3867']
ah_corr (numeric, 540 distinct): ['0.4283', '0.7202', '0.0964', '0.2915', '0.0076', '0.2708', '0.7843', '0.0284', '0.7723', '0.5933']
ah_bolus (numeric, 540 distinct): ['0.5679', '0.2534', '0.1045', '0.2836', '0.8479', '0.3229', '0.1385', '0.1262', '0.9422', '0.8816']
slm_corr (numeric, 540 distinct): ['0.4744', '0.4348', '0.4432', '0.5482', '0.5832', '0.3278', '0.8947', '0.9224', '0.3794', '0.622']
efficiency_factor (numeric, 540 distinct): ['0.2457', '0.2262', '0.2666', '0.6206', '0.1692', '0.3838', '0.1115', '0.6495', '0.127', '0.8367']
tidal_mix_max (numeric, 540 distinct): ['0.1042', '0.5448', '0.546', '0.6445', '0.9048', '0.691', '0.3202', '0.2816', '0.4768', '0.2572']
vertical_decay_scale (numeric, 540 distinct): ['0.8691', '0.9937', '0.9622', '0.7597', '0.2629', '0.7854', '0.4637', '0.8707', '0.9973', '0.459']
convect_corr (numeric, 540 distinct): ['0.9975', '0.8842', '0.2865', '0.7158', '0.9001', '0.5169', '0.9659', '0.6405', '0.1633', '0.401']
bckgrnd_vdc1 (numeric, 540 distinct): ['0.4486', '0.3142', '0.0549', '0.1926', '0.6657', '0.39', '0.0199', '0.8274', '0.9521', '0.0393']
bckgrnd_vdc_ban (numeric, 540 distinct): ['0.3075', '0.9006', '0.639', '0.8496', '0.6974', '0.1094', '0.2499', '0.2239', '0.8199', '0.5919']
bckgrnd_vdc_eq (numeric, 540 distinct): ['0.8583', '0.176', '0.6689', '0.3059', '0.6366', '0.5605', '0.2535', '0.6505', '0.9523', '0.4324']
bckgrnd_vdc_psim (numeric, 540 distinct): ['0.797', '0.491', '0.6681', '0.2771', '0.03', '0.101', '0.6214', '0.7999', '0.4941', '0.4074']
Prandtl (numeric, 540 distinct): ['0.8699', '0.3342', '0.0329', '0.404', '0.745', '0.1626', '0.8974', '0.2296', '0.6075', '0.1758']
'''

CONTEXT = "Climate Model Simulation Crashes"
TARGET = CuratedTarget(raw_name="outcome", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "success", "0": "failure"})
COLS_TO_DROP = []
FEATURES = []