from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: eeg-eye-state
====
Examples: 14980
====
URL: https://www.openml.org/search?type=data&id=1471
====
Description: **Author**: Oliver Roesler  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State), Baden-Wuerttemberg, Cooperative State University (DHBW), Stuttgart, Germany  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  

All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analyzing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.

The features correspond to 14 EEG measurements from the headset, originally labeled AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4, in that order.
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 548 distinct): ['4291.79', '4287.69', '4295.9', '4292.31', '4291.28', '4294.36', '4297.95', '4296.92', '4280.51', '4289.23']
V2 (numeric, 452 distinct): ['4003.59', '4007.18', '4006.67', '4003.08', '4008.72', '4007.69', '4005.13', '4000.51', '4002.05', '4004.1']
V3 (numeric, 345 distinct): ['4263.59', '4264.1', '4262.56', '4265.13', '4264.62', '4263.08', '4262.05', '4261.54', '4265.64', '4261.03']
V4 (numeric, 312 distinct): ['4122.56', '4121.54', '4120.51', '4122.05', '4121.03', '4123.08', '4120.0', '4108.21', '4107.18', '4106.67']
V5 (numeric, 285 distinct): ['4332.31', '4334.87', '4336.41', '4335.38', '4334.36', '4336.92', '4335.9', '4333.85', '4332.82', '4343.59']
V6 (numeric, 330 distinct): ['4616.41', '4616.92', '4615.9', '4615.38', '4617.95', '4617.44', '4614.87', '4618.46', '4614.36', '4618.97']
V7 (numeric, 290 distinct): ['4072.31', '4071.28', '4075.38', '4070.26', '4056.92', '4067.18', '4072.82', '4070.77', '4071.79', '4055.38']
V8 (numeric, 294 distinct): ['4610.77', '4612.31', '4612.82', '4611.79', '4609.74', '4615.38', '4614.36', '4613.85', '4614.87', '4613.33']
V9 (numeric, 304 distinct): ['4196.92', '4206.15', '4199.49', '4195.9', '4192.31', '4194.87', '4193.85', '4197.44', '4193.33', '4194.36']
V10 (numeric, 346 distinct): ['4224.62', '4229.23', '4225.13', '4228.21', '4222.56', '4226.67', '4227.69', '4227.18', '4226.15', '4230.26']
V11 (numeric, 419 distinct): ['4195.38', '4195.9', '4208.21', '4207.69', '4194.36', '4204.62', '4207.18', '4192.82', '4197.44', '4208.72']
V12 (numeric, 343 distinct): ['4273.85', '4271.28', '4270.26', '4272.31', '4274.87', '4273.33', '4279.49', '4271.79', '4272.82', '4276.92']
V13 (numeric, 558 distinct): ['4603.08', '4606.15', '4596.41', '4603.59', '4605.64', '4592.31', '4609.74', '4604.1', '4602.05', '4609.23']
V14 (numeric, 592 distinct): ['4352.31', '4351.28', '4352.82', '4354.36', '4354.87', '4349.74', '4355.9', '4347.69', '4351.79', '4350.77']
'''

EEG_MEASURES = ["AF3", "F7", "F3", "FC5", "T7", "P", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CONTEXT = "Eye State based on EEG Mesurements"
TARGET = CuratedTarget(raw_name="Class", new_name="Eye State", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=f"V{i}", new_name=f"EEG Measure {eeg}")
            for i, eeg in enumerate(EEG_MEASURES, start=1)]