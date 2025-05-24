from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: mimic_extract_los_3
====
Examples: 4156450
====
URL: https://www.openml.org/search?type=data&id=46678
====
Description: Length of hospital stay is, along with patient mortality, ''the most important clinical outcome'' for an ICU admission. Accurately predicting the length of stay of a patient can aid in assessment of the severity of a patient's condition. Of particular clinical relevance, making these predictions early and with a non-zero time gap*between the prediction and the outcome is of real-world importance: predictions must be made sufficiently early such that a patient's treatment can be adjusted to potentially avoid a negative outcome. The importance of this prediction task for real-world clinical care is underscored by the many previous works in the medical literature addressing this prediction topic (see our paper for references).
Here only the table PRESCRIPTIONS is used. 
The multiclass ROUTE column typically refer to the route of drug administration. Here's what those abbreviations commonly mean in a clinical context:
PO: By Mouth (Oral)
IV: Intravenous
IV DRIP: Intravenous Drip
SC: Subcutaneous
SL: Sublingual
IM: Intramuscular
NG: Nasogastric Tube
PR: Per Rectum
OU: Both Eyes (Oculus Uterque)
IVPCA: Intravenous Patient-Controlled Analgesia
IH: Inhalation
TP: Topical
DIALYS: Dialysis
NU: Nurse Administered
ORAL: Oral (By Mouth)
IV BOLUS: Intravenous Bolus
PO/NG: Oral or Nasogastric Tube
TD: Transdermal
BOTH EYES: Same as OU
PB: Piggyback IV
ED: Epidural
REPLACE: Replacement Therapy
G TUBE: Gastrostomy Tube
ET: Endotracheal
DWELL: Dwell Time (e.g., in peritoneum)
IR: Irrigation
VG: Vaginal
LEFT EYE: Oculus Sinister (OS)
NEB: Nebulization
IN: Intranasal
ID: Intradermal
AU: Both Ears (Auris Uterque)
OS: Left Eye
IRR: Irrigation
OD: Right Eye
INHALATION: Same as IH
IA: Intra-Arterial
AS: Left Ear (Auris Sinistra)
AD: Right Ear (Auris Dextra)
J TUBE: Jejunostomy Tube
LOCK: IV Lock
IT: Intrathecal
IJ: Intrajugular
IP: Intraperitoneal
PL: Placenta Administration
LEFT EAR: Same as AS
NAS: Nasal
TT: Transtracheal
RIGHT EYE: Oculus Dexter (OD)
EX-VIVO: Outside the Body
NS: Normal Saline
BU: Buccal
BOTH EARS: Same as AU
BUCCAL: Same as BU
SUBCUT: Subcutaneous (SC)
SCPUMP: Subcutaneous Pump
ND: Nasoduodenal
PERIPHNERVE: Peripherally Inserted Nerve
AERO: Aerosol
IO: Intraosseous
VT: Ventilation Tube
RIGHT EAR: Same as AD
PO/IV: Oral or Intravenous
OG: Orogastric Tube
PO/PR: Oral or Rectal
RECTAL: Same as PR
SCPCA: Subcutaneous Patient-Controlled Analgesia
IVT: Intravitreal
IC: Intracardiac
IVS: Intravenous Subcutaneously
NG/OG: Nasogastric or Orogastric Tube
ENTERAL TUBE ONLY: Not Oral, via Tube
PO OR ENTERAL TUBE: Oral or Tube
PO/OG: Oral or Orogastric Tube
INTERSPACE: Between Spaces (e.g., vertebrae)
INTRAPERICARDIAL: Into the Pericardium
LUMBAR PLEXUS: Nerve Plexus in the Lumbar Spine
AXILLARY: Axillary Region (e.g., underarm)

This dataset was obtained from Kaggle: https://www.kaggle.com/hussameldinanwer/mimic-iii
====
Target Variable: ROUTE (string, 78 distinct): ['IV', 'PO', 'IV DRIP', 'PO/NG', 'SC', 'IH', 'IM', 'PR', 'NG', 'TP']
====
Features:

ROW_ID (numeric, 4156450 distinct): ['2214776', '1096881', '1096131', '1096132', '1096134', '1096135', '1096136', '1096120', '1096876', '1096892']
SUBJECT_ID (numeric, 39363 distinct): ['29035', '11318', '13033', '109', '19213', '25225', '25256', '7809', '11861', '48872']
HADM_ID (numeric, 50216 distinct): ['131118', '101936', '157559', '168201', '194773', '123178', '109520', '115396', '129611', '107543']
ICUSTAY_ID (numeric, 52151 distinct): ['275498.0', '233352.0', '215683.0', '255875.0', '241307.0', '290964.0', '257336.0', '286492.0', '290931.0', '245503.0']
STARTDATE (string, 38497 distinct): ['2182-12-04 00:00:00', '2132-07-30 00:00:00', '2187-08-23 00:00:00', '2128-05-25 00:00:00', '2172-09-30 00:00:00', '2143-11-20 00:00:00', '2151-06-29 00:00:00', '2115-02-26 00:00:00', '2164-10-23 00:00:00', '2126-05-07 00:00:00']
ENDDATE (string, 38500 distinct): ['2159-01-23 00:00:00', '2156-12-22 00:00:00', '2176-03-20 00:00:00', '2177-11-18 00:00:00', '2104-03-26 00:00:00', '2113-11-28 00:00:00', '2108-02-23 00:00:00', '2133-02-10 00:00:00', '2166-04-04 00:00:00', '2106-01-12 00:00:00']
DRUG_TYPE (string, 3 distinct): ['MAIN', 'BASE', 'ADDITIVE']
DRUG (string, 4525 distinct): ['Potassium Chloride', 'Insulin', 'D5W', 'Furosemide', '0.9% Sodium Chloride', 'NS', 'Magnesium Sulfate', 'Iso-Osmotic Dextrose', 'Sodium Chloride 0.9%  Flush', 'Acetaminophen']
DRUG_NAME_POE (string, 4036 distinct): ['Insulin', 'Furosemide', 'Potassium Chloride', 'Sodium Chloride 0.9%  Flush', 'Acetaminophen', 'Metoprolol', 'Metoprolol Tartrate', 'Morphine Sulfate', 'Lorazepam', 'Heparin']
DRUG_NAME_GENERIC (string, 2863 distinct): ['Furosemide', 'Potassium Chloride', 'Sodium Chloride 0.9%  Flush', 'Metoprolol', 'Insulin - Sliding Scale', 'Acetaminophen', 'Metoprolol Tartrate', 'Lorazepam', 'Heparin Sodium', 'Docusate Sodium']
FORMULARY_DRUG_CD (string, 3267 distinct): ['FURO40I', 'NACLFLUSH', 'INSULIN', 'D5W250', 'NS1000', 'NS500', 'MAG2PM', 'VANC1F', 'VANCOBASE', 'METO25']
GSN (string, 4686 distinct): ['nan', '001210', '001972', '008205', '001723', '016546', '006549', '050631', '004489', '045309']
NDC (numeric, 4204 distinct): ['0.0', '338001702.0', '338004904.0', '338004903.0', '409672924.0', '517570425.0', '51079025520.0', '338004902.0', '338070341.0', '338355248.0']
PROD_STRENGTH (string, 4000 distinct): ['250mL Bag', '1000mL Bag', '100mL Bag', 'Syringe', '40mg/4mL Vial', '50ml Bag', '500mL Bag', 'Dummy Package for Sliding Scale', '50 mL Bag', '25mg Tablet']
DOSE_VAL_RX (string, 2605 distinct): ['100', '1', '1000', '250', '40', '2', '50', '20', '10', '500']
DOSE_UNIT_RX (string, 104 distinct): ['mg', 'mL', 'ml', 'UNIT', 'mEq', 'gm', 'TAB', 'g', 'mcg', 'NEB']
FORM_VAL_DISP (string, 3073 distinct): ['1', '2', '0.5', '250', '100', '50', '1-2', '4', '0', '0.6']
FORM_UNIT_DISP (string, 84 distinct): ['TAB', 'VIAL', 'BAG', 'ml', 'mL', 'SYR', 'CAP', 'UDCUP', 'BTL', 'PKT']
'''

CONTEXT = "MIMIC Route of Drug Administration"
TARGET = CuratedTarget(raw_name="ROUTE", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["ROW_ID"]
FEATURES = [CuratedFeature(raw_name="STARTDATE", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="ENDDATE", feat_type=FeatureType.DATE),]