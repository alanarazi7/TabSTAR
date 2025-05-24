from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: drug-directory
====
Examples: 120215
====
URL: https://www.openml.org/search?type=data&id=43044
====
Description: Product listing data submitted to the U.S. FDA for all unfinished, unapproved drugs.
====
Target Variable: PRODUCTTYPENAME (nominal, 7 distinct): ['HUMAN OTC DRUG', 'HUMAN PRESCRIPTION DRUG', 'NON-STANDARDIZED ALLERGENIC', 'PLASMA DERIVATIVE', 'STANDARDIZED ALLERGENIC', 'VACCINE', 'CELLULAR THERAPY']
====
Features:

PRODUCTID (string, 120215 distinct): ['0002-0800_03155cd7-d92b-480b-8a86-087659b6990f', '68071-5273_a7586104-9e0c-ef22-e053-2a95a90ace27', '68071-5285_a8623225-de0d-28c6-e053-2995a90ad709', '68071-5284_a861cf7d-1aff-7e5d-e053-2995a90ad14e', '68071-5283_a860ada6-b230-acb8-e053-2a95a90a591c', '68071-5282_a84a00ab-6b70-a860-e053-2995a90a7d1b', '68071-5281_a7e575de-2dfa-6518-e053-2995a90ab34f', '68071-5280_a7d53a1d-5c6f-1c1d-e053-2a95a90a40d1', '68071-5279_a7d15f71-3052-5452-e053-2a95a90a1d4e', '68071-5278_a822a86b-c870-7c2a-e053-2995a90af296']
PRODUCTNDC (string, 117896 distinct): ['54365-400', '72728-002', '72766-002', '78888-080', '75772-420', '70483-001', '75323-180', '76943-081', '75674-425', '61919-537']
PROPRIETARYNAME (string, 45019 distinct): ['Hand Sanitizer', 'Ibuprofen', 'Oxygen', 'HAND SANITIZER', 'Gabapentin', 'Allergy Relief', 'Aripiprazole', 'Amoxicillin', 'Levothyroxine Sodium', 'Lamotrigine']
PROPRIETARYNAMESUFFIX (string, 4569 distinct): ['Maximum Strength', 'Extra Strength', '01', 'Extended-Release', 'Regular Strength', 'Childrens', 'XR', 'ER', 'Nighttime', 'HP']
NONPROPRIETARYNAME (string, 19307 distinct): ['ALCOHOL', 'Alcohol', 'Ethyl Alcohol', 'Isopropyl Alcohol', 'Benzalkonium Chloride', 'Ibuprofen', 'BENZALKONIUM CHLORIDE', 'Acetaminophen', 'Menthol', 'Hand Sanitizer']
DOSAGEFORMNAME (string, 139 distinct): ['TABLET', 'LIQUID', 'GEL', 'TABLET, FILM COATED', 'CREAM', 'SOLUTION', 'CAPSULE', 'INJECTION, SOLUTION', 'PELLET', 'LOTION']
ROUTENAME (string, 192 distinct): ['ORAL', 'TOPICAL', 'INTRAVENOUS', 'SUBCUTANEOUS', 'RESPIRATORY (INHALATION)', 'INTRADERMAL; PERCUTANEOUS; SUBCUTANEOUS', 'OPHTHALMIC', 'DENTAL', 'INTRAMUSCULAR; INTRAVENOUS', 'EXTRACORPOREAL']
STARTMARKETINGDATE (numeric, 7474 distinct): ['20200330', '19830303', '19810915', '20090901', '19650101', '20200401', '20200601', '20200501', '20200701', '20190101']
ENDMARKETINGDATE (numeric, 676 distinct): ['20201231.0', '20211231.0', '20211230.0', '20210331.0', '20210228.0', '20221230.0', '20210531.0', '20210430.0', '20210131.0', '20210930.0']
MARKETINGCATEGORYNAME (string, 10 distinct): ['ANDA', 'OTC MONOGRAPH NOT FINAL', 'OTC MONOGRAPH FINAL', 'UNAPPROVED HOMEOPATHIC', 'NDA', 'BLA', 'UNAPPROVED DRUG OTHER', 'NDA AUTHORIZED GENERIC', 'UNAPPROVED MEDICAL GAS', 'UNAPPROVED DRUG FOR USE IN DRUG SHORTAGE']
APPLICATIONNUMBER (string, 11256 distinct): ['part333A', 'part352', 'part333E', 'part341', 'part348', 'part343', 'part334', 'part347', 'BLA101833', 'BLA103753']
LABELERNAME (string, 13388 distinct): ['Bryant Ranch Prepack', 'Boiron', 'REMEDYREPACK INC.', 'A-S Medication Solutions', 'Washington Homeopathic Products', 'Proficient Rx LP', 'NuCare Pharmaceuticals,Inc.', 'Greer Laboratories, Inc.', 'ALK-Abello, Inc.', 'Uriel Pharmacy Inc.']
SUBSTANCENAME (string, 9729 distinct): ['ALCOHOL', 'BENZALKONIUM CHLORIDE', 'ISOPROPYL ALCOHOL', 'IBUPROFEN', 'ACETAMINOPHEN', 'SALICYLIC ACID', 'MENTHOL', 'SODIUM FLUORIDE', 'DIPHENHYDRAMINE HYDROCHLORIDE', 'ZINC OXIDE']
ACTIVE_NUMERATOR_STRENGTH (string, 10204 distinct): ['10', '70', '30', '1', '100', '75', '5', '20', '50', '80']
ACTIVE_INGRED_UNIT (string, 2927 distinct): ['mg/1', 'mL/100mL', 'mg/mL', 'mg/1; mg/1', 'g/100g', 'mg/g', 'g/100mL', 'g/mL', 'mL/mL', 'mg/5mL']
PHARM_CLASSES (string, 1319 distinct): ['Corticosteroid [EPC],Corticosteroid Hormone Receptor Agonists [MoA]', 'Atypical Antipsychotic [EPC]', 'Cyclooxygenase Inhibitors [MoA],Anti-Inflammatory Agents, Non-Steroidal [CS],Nonsteroidal Anti-inflammatory Drug [EPC]', 'Full Opioid Agonists [MoA],Opioid Agonist [EPC]', 'Non-Standardized Pollen Allergenic Extract [EPC],Increased Histamine Release [PE],Cell-mediated Immunity [PE],Increased IgG Production [PE],Pollen [CS],Allergens [CS]', 'Adrenergic beta-Antagonists [MoA],beta-Adrenergic Blocker [EPC]', 'HMG-CoA Reductase Inhibitor [EPC],Hydroxymethylglutaryl-CoA Reductase Inhibitors [MoA]', 'Serotonin Reuptake Inhibitor [EPC],Serotonin Uptake Inhibitors [MoA]', 'Benzodiazepine [EPC],Benzodiazepines [CS]', 'Angiotensin 2 Receptor Antagonists [MoA],Angiotensin 2 Receptor Blocker [EPC]']
DEASCHEDULE (string, 4 distinct): ['CII', 'CIV', 'CIII', 'CV']
NDC_EXCLUDE_FLAG (string, 1 distinct): ['N']
LISTING_RECORD_CERTIFIED_THROUGH (numeric, 2 distinct): ['20211231.0', '20201231.0']
'''

CONTEXT = "Product listing submitted to the US FDA for unfinished, unapproved drugs"
TARGET = CuratedTarget(raw_name="PRODUCTTYPENAME", new_name="PRODUCT TYPE NAME",
                       task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["PRODUCTID", "PRODUCTNDC"]
FEATURES = []