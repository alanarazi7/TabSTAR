from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: bone_marrow_transplant_children
====
Examples: 187
====
URL: https://www.openml.org/search?type=data&id=46610
====
Description: Dataset Information
Additional Information

The data set describes pediatric patients with several hematologic diseases: malignant disorders (i.a. acute lymphoblastic leukemia, acute myelogenous leukemia, chronic myelogenous leukemia, myelodysplastic syndrome) and nonmalignant cases (i.a. severe aplastic anemia, Fanconi anemia, with X-linked adrenoleukodystrophy). All patients were subject to the unmanipulated allogeneic unrelated donor hematopoietic stem cell transplantation. 
 
The motivation of the study was to identify the most important factors influencing the success or failure of the transplantation procedure. In particular, the aim was to verify the hypothesis that increased dosage of CD34+ cells / kg extends overall survival time without simultaneous occurrence of undesirable events affecting patients' quality of life (Ka wah, et al., 2010).

The data set has been used in our work concerning survival rules (Wrobel et al., 2017) and user-guided rule induction (Sikora et al., 2019). The authors of the research on stem cell transplantation (Ka wah, et al., 2010) who inspired our study also contributed to the set.

Has Missing Values?

Yes 


_______________________


donor_age - Age of the donor at the time of hematopoietic stem cells apheresis
donor_age_below_35 - Is donor age less than 35 (yes, no)
donor_ABO - ABO blood group of the donor of hematopoietic stem cells (0, A, B, AB)
donor_CMV - Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (present, absent)
recipient_age - Age of the recipient of hematopoietic stem cells at the time of transplantation
recipient_age_below_10 - Is recipient age below 10 (yes, no)
recipient_age_int - Age of the recipient discretized to intervals (0,5], (5, 10], (10, 20]
recipient_gender - Gender of the recipient (female, male)
recipient_body_mass - Body mass of the recipient of hematopoietic stem cells at the time of the transplantation
recipient_ABO - ABO blood group of the recipient of hematopoietic stem cells (0, A, B, AB)
recipient_rh - Presence of the Rh factor on recipients red blood cells (plus, minus)
recipient_CMV - Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (present, absent)
disease - Type of disease (ALL, AML, chronic, nonmalignant, lymphoma)
disease_group - Type of disease (malignant, nonmalignant)
gender_match - Compatibility of the donor and recipient according to their gender (female to male, other)
ABO_match - Compatibility of the donor and the recipient of hematopoietic stem cells according to ABO blood group (matched, mismatched)
CMV_status - Serological compatibility of the donor and the recipient of hematopoietic stem cells according to cytomegalovirus infection prior to transplantation (the higher the value, the lower the compatibility)
HLA_match - Compatibility of antigens of the main histocompatibility complex of the donor and the recipient of hematopoietic stem cells (10/10, 9/10, 8/10, 7/10)
HLA_mismatch - HLA matched or mismatched
antigen - In how many antigens there is a difference between the donor and the recipient (0-3)
allel - In how many allele there is a difference between the donor and the recipient (0-4)
HLA_group_1 - The difference type between the donor and the recipient (HLA matched, one antigen, one allel, DRB1 cell, two allele or allel+antigen, two antigenes+allel, mismatched)
risk_group - Risk group (high, low)
stem_cell_source - Source of hematopoietic stem cells (peripheral blood, bone marrow)
tx_post_relapse - The second bone marrow transplantation after relapse (yes ,no)
CD34_x1e6_per_kg - CD34kgx10d6 - CD34+ cell dose per kg of recipient body weight (10^6/kg)
CD3_x1e8_per_kg - CD3+ cell dose per kg of recipient body weight (10^8/kg)
CD3_to_CD34_ratio - CD3+ cell to CD34+ cell ratio
ANC_recovery - Neutrophils recovery defined as neutrophils count >0.5 x 10^9/L (yes, no)
time_to_ANC_recovery - Time in days to neutrophils recovery
PLT_recovery - Platelet recovery defined as platelet count >50000/mm3 (yes, no)
time_to_PLT_recovery - Time in days to platelet recovery
acute_GvHD_II_III_IV - Development of acute graft versus host disease stage II or III or IV (yes, no)
acute_GvHD_III_IV - Development of acute graft versus host disease stage III or IV (yes, no)
time_to_acute_GvHD_III_IV - Time in days to development of acute graft versus host disease stage III or IV
extensive_chronic_GvHD - Development of extensive chronic graft versus host disease (yes, no)
relapse - Relapse of the disease (yes, no)
survival_time - Time of observation (if alive) or time to event (if dead) in days
survival_status - Survival status (0 - alive, 1 - dead)
====
Target Variable: survival_status (numeric, 2 distinct): ['0', '1']
====
Features:

Recipientgender (numeric, 2 distinct): ['1', '0']
Stemcellsource (numeric, 2 distinct): ['1', '0']
Donorage (numeric, 187 distinct): ['22.8301', '47.3699', '25.2849', '33.6438', '33.8575', '24.2849', '51.9918', '29.1589', '36.4356', '37.3808']
Donorage35 (numeric, 2 distinct): ['0', '1']
IIIV (numeric, 2 distinct): ['1', '0']
Gendermatch (numeric, 2 distinct): ['0', '1']
DonorABO (numeric, 4 distinct): ['0', '1', '-1', '2']
RecipientABO (numeric, 4 distinct): ['1.0', '-1.0', '0.0', '2.0']
RecipientRh (numeric, 2 distinct): ['1.0', '0.0']
ABOmatch (numeric, 2 distinct): ['1.0', '0.0']
CMVstatus (numeric, 4 distinct): ['2.0', '0.0', '3.0', '1.0']
DonorCMV (numeric, 2 distinct): ['0.0', '1.0']
RecipientCMV (numeric, 2 distinct): ['1.0', '0.0']
Disease (string, 5 distinct): ['ALL', 'chronic', 'AML', 'nonmalignant', 'lymphoma']
Riskgroup (numeric, 2 distinct): ['0', '1']
Txpostrelapse (numeric, 2 distinct): ['0', '1']
Diseasegroup (numeric, 2 distinct): ['1', '0']
HLAmatch (numeric, 4 distinct): ['0', '1', '2', '3']
HLAmismatch (numeric, 2 distinct): ['0', '1']
Antigen (numeric, 4 distinct): ['-1.0', '1.0', '0.0', '2.0']
Allele (numeric, 5 distinct): ['-1.0', '0.0', '1.0', '2.0', '3.0']
HLAgrI (numeric, 7 distinct): ['0', '1', '4', '2', '3', '7', '5']
Recipientage (numeric, 125 distinct): ['17.8', '13.5', '11.5', '13.4', '8.5', '5.0', '3.4', '14.0', '12.7', '3.1']
Recipientage10 (numeric, 2 distinct): ['0', '1']
Recipientageint (numeric, 3 distinct): ['2', '1', '0']
Relapse (numeric, 2 distinct): ['0', '1']
aGvHDIIIIV (numeric, 2 distinct): ['1', '0']
extcGvHD (numeric, 2 distinct): ['1.0', '0.0']
CD34kgx10d6 (numeric, 183 distinct): ['7.2', '3.53', '12.58', '5.37', '7.78', '7.97', '10.75', '4.37', '4.44', '7.45']
CD3dCD34 (numeric, 182 distinct): ['1.3388', '5.0458', '1.018', '4.489', '1.8514', '4.2187', '5.1646', '1.0703', '1.2834', '1.0829']
CD3dkgx10d8 (numeric, 163 distinct): ['7.32', '0.13', '0.4', '5.8', '5.64', '4.94', '2.14', '8.06', '2.22', '5.16']
Rbodymass (numeric, 130 distinct): ['33.0', '23.0', '15.0', '24.0', '62.0', '49.0', '65.0', '47.0', '13.0', '30.0']
ANCrecovery (numeric, 18 distinct): ['15', '16', '14', '13', '17', '12', '18', '11', '1000000', '19']
PLTrecovery (numeric, 50 distinct): ['1000000', '14', '21', '16', '13', '17', '19', '24', '12', '20']
time_to_aGvHD_III_IV (numeric, 28 distinct): ['1000000', '18', '21', '14', '19', '11', '16', '100', '15', '42']
survival_time (numeric, 174 distinct): ['60', '41', '385', '676', '1243', '48', '28', '1895', '149', '1754']
'''

CONTEXT = "Bone Marrow Transplant Children: Survival Analysis"
TARGET = CuratedTarget(raw_name="survival_status", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []