from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: physionet_sepsis
====
Examples: 1552210
====
URL: https://www.openml.org/search?type=data&id=46677
====
Description: Sepsis is a life-threatening condition that occurs when the body's response to infection causes tissue damage, organ failure, or death (Singer et al., 2016). In the U.S., nearly 1.7 million people develop sepsis and 270,000 people die from sepsis each year; over one third of people who die in U.S. hospitals have sepsis (CDC). Internationally, an estimated 30 million people develop sepsis and 6 million people die from sepsis each year; an estimated 4.2 million newborns and children are affected (WHO). Sepsis costs U.S. hospitals more than any other health condition at $24 billion (13 percentage of U.S. healthcare expenses) a year, and a majority of these costs are for sepsis patients that were not diagnosed at admission (Paoli et al., 2018). Sepsis costs are even greater globally with the developing world at most risk. Altogether, sepsis is a major public health issue responsible for significant morbidity, mortality, and healthcare expenses.

Early detection and antibiotic treatment of sepsis are critical for improving sepsis outcomes, where each hour of delayed treatment has been associated with roughly an 4-8 percentage increase in mortality (Kumar et al., 2006; Seymour et al., 2017). To help address this problem, clinicians have proposed new definitions for sepsis (Singer et al., 2016), but the fundamental need to detect and treat sepsis early still remains, and basic questions about the limits of early detection remain unanswered. The PhysioNet/Computing in Cardiology Challenge 2019 provides an opportunity to address these questions.

Data column name transformation and mapping are coming from https://github.com/mlfoundations/tableshift/tree/fca9429814703a07e3902d005d46563a207b7f0a
Columns names are restricted to less than 64 characters and added a counter to avoid duplicates for the first characters.
====
Target Variable: SepsisLabel (numeric, 2 distinct): ['0', '1']
====
Features:

Unnamed__0 (numeric, 336 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Hour (numeric, 336 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Heart_rate__in_beats_per_minute (numeric, 343 distinct): ['80.0', '90.0', '88.0', '82.0', '84.0', '78.0', '70.0', '86.0', '76.0', '74.0']
Pulse_oximetry__percentage (numeric, 145 distinct): ['100.0', '98.0', '99.0', '97.0', '96.0', '95.0', '94.0', '93.0', '92.0', '99.5']
Temperature__deg_C (numeric, 641 distinct): ['37.0', '36.5', '36.8', '37.2', '37.5', '36.7', '36.6', '37.4', '37.1', '37.3']
Systolic_BP__mm_Hg (numeric, 990 distinct): ['116.0', '112.0', '110.0', '114.0', '118.0', '120.0', '122.0', '108.0', '119.0', '124.0']
Mean_arterial_pressure__mm_Hg (numeric, 945 distinct): ['78.0', '76.0', '74.0', '80.0', '72.0', '82.0', '75.0', '77.0', '79.0', '84.0']
Diastolic_BP__mm_Hg (numeric, 678 distinct): ['58.0', '60.0', '56.0', '62.0', '59.0', '54.0', '64.0', '57.0', '55.0', '63.0']
Respiration_rate__breaths_per_minute (numeric, 235 distinct): ['18.0', '16.0', '20.0', '14.0', '17.0', '15.0', '22.0', '19.0', '21.0', '12.0']
End_tidal_carbon_dioxide__mm_Hg (numeric, 139 distinct): ['35.0', '34.0', '32.0', '33.0', '31.0', '30.0', '36.0', '37.0', '29.0', '38.0']
Excess_bicarbonate__mmol_L (numeric, 407 distinct): ['0.0', '-1.0', '-2.0', '1.0', '-3.0', '2.0', '-4.0', '3.0', '-5.0', '4.0']
Bicarbonate__mmol_L (numeric, 304 distinct): ['24.0', '25.0', '23.0', '26.0', '22.0', '27.0', '21.0', '28.0', '20.0', '29.0']
Fraction_of_inspired_oxygen__percentage (numeric, 104 distinct): ['0.4', '0.5', '1.0', '0.6', '0.7', '0.35', '0.3', '0.8', '0.21', '0.45']
pH (numeric, 103 distinct): ['7.4', '7.38', '7.39', '7.41', '7.36', '7.42', '7.37', '7.35', '7.43', '7.34']
Partial_pressure_of_carbon_dioxide_from_arterial_blood__mm_Hg (numeric, 551 distinct): ['38.0', '40.0', '39.0', '42.0', '37.0', '36.0', '41.0', '43.0', '35.0', '44.0']
Oxygen_saturation_from_arterial_blood__percentage (numeric, 432 distinct): ['98.0', '97.0', '96.0', '99.0', '95.0', '94.0', '93.0', '92.0', '98.5', '98.8']
Aspartate_transaminase__IU_L (numeric, 2025 distinct): ['17.0', '20.0', '18.0', '19.0', '21.0', '16.0', '22.0', '24.0', '23.0', '15.0']
Blood_urea_nitrogen__mg_dL (numeric, 265 distinct): ['14.0', '13.0', '12.0', '11.0', '15.0', '10.0', '16.0', '17.0', '9.0', '18.0']
Alkaline_phosphatase__IU_L (numeric, 752 distinct): ['58.0', '49.0', '56.0', '54.0', '53.0', '55.0', '52.0', '61.0', '50.0', '59.0']
Calcium__mg_dL (numeric, 549 distinct): ['8.3', '8.4', '8.5', '8.2', '8.6', '8.1', '8.7', '8.0', '8.8', '7.9']
Chloride__mmol_L (numeric, 108 distinct): ['106.0', '107.0', '105.0', '108.0', '104.0', '109.0', '103.0', '110.0', '102.0', '111.0']
Creatinine__mg_dL (numeric, 1407 distinct): ['0.7', '0.8', '0.6', '0.9', '1.0', '0.5', '1.1', '1.2', '1.3', '0.4']
Direct_bilirubin__mg_dL (numeric, 280 distinct): ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '1.0', '0.8', '0.9']
Serum_glucose__mg_dL (numeric, 1157 distinct): ['118.0', '114.0', '116.0', '120.0', '121.0', '112.0', '115.0', '117.0', '119.0', '113.0']
Lactic_acid__mg_dL (numeric, 1340 distinct): ['1.2', '1.0', '1.3', '1.4', '1.1', '0.9', '1.6', '1.5', '1.7', '0.8']
Magnesium__mmol_dL (numeric, 110 distinct): ['2.0', '1.9', '2.1', '1.8', '2.2', '1.7', '2.3', '2.4', '1.6', '2.5']
Phosphate__mg_dL (numeric, 193 distinct): ['3.2', '2.9', '3.1', '3.3', '3.4', '3.0', '2.8', '3.5', '2.7', '3.6']
Potassium__mmol_L (numeric, 402 distinct): ['4.0', '3.9', '4.1', '3.8', '4.2', '3.7', '4.3', '4.4', '3.6', '4.5']
Total_bilirubin__mg_dL (numeric, 407 distinct): ['0.5', '0.6', '0.4', '0.7', '0.8', '0.3', '0.9', '1.0', '1.1', '0.2']
Troponin_I__ng_mL (numeric, 2423 distinct): ['0.01', '0.03', '0.02', '0.04', '0.05', '0.06', '0.07', '40.0', '0.08', '0.1']
Hematocrit__percentage (numeric, 725 distinct): ['29.0', '28.0', '30.0', '32.0', '31.0', '26.0', '27.0', '33.0', '34.0', '29.1']
Hemoglobin__g_dL (numeric, 340 distinct): ['10.0', '10.1', '10.3', '10.2', '9.9', '9.7', '10.5', '9.8', '9.4', '10.7']
Partial_thromboplastin_time__seconds (numeric, 1410 distinct): ['150.0', '27.7', '28.1', '29.8', '28.5', '28.6', '28.8', '27.9', '28.7', '29.2']
Leukocyte_count__count_L (numeric, 891 distinct): ['8.6', '8.8', '10.0', '9.4', '9.8', '9.0', '9.2', '10.2', '8.0', '8.2']
Fibrinogen_concentration__mg_dL (numeric, 823 distinct): ['217.0', '180.0', '202.0', '185.0', '219.0', '200.0', '151.0', '214.0', '248.0', '208.0']
Platelet_count__count_mL (numeric, 989 distinct): ['167.0', '162.0', '158.0', '175.0', '166.0', '187.0', '159.0', '149.0', '186.0', '152.0']
Age__years (numeric, 5987 distinct): ['67.0', '68.0', '65.0', '71.0', '66.0', '61.0', '69.0', '62.0', '73.0', '70.0']
Female__0__or_male__1 (numeric, 2 distinct): ['1', '0']
Administrative_identifier_for_ICU_unit__MICU___false__0__or_true (numeric, 2 distinct): ['0.0', '1.0']
Administrative_identifier_for_ICU_unit__SICU___false__0__or_true (numeric, 2 distinct): ['1.0', '0.0']
Time_between_hospital_and_ICU_admission__hours_since_ICU_admissi (numeric, 12156 distinct): ['-0.02', '-0.03', '0.0', '-0.01', '-0.04', '-0.05', '-0.06', '-0.07', '-0.09', '-0.08']
ICU_length_of_stay__hours_since_ICU_admission (numeric, 336 distinct): ['8', '7', '9', '6', '10', '11', '12', '5', '13', '14']
Patient_ID (numeric, 40336 distinct): ['3658', '114471', '101922', '117406', '4905', '18469', '113190', '16581', '8132', '116439']
'''

CONTEXT = "Sepsis Prediction"
TARGET = CuratedTarget(raw_name="SepsisLabel", new_name="Sepsis Status", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["Unnamed__0"]
FEATURES = []