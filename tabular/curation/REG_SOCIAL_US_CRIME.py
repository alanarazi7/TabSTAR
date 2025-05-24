from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: us_crime
====
Examples: 1994
====
URL: https://www.openml.org/search?type=data&id=42730
====
Description: Ignores community name.**Author**:   
  
**Source**: Unknown - 2009  
**Please cite**:   

Title: Communities and Crime
 
Abstract: Communities within the United States. The data combines socio-economic data from the 1990 US Census, law enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR.

Data Set Characteristics:  Multivariate
Attribute Characteristics: Real
Associated Tasks: Regression
Number of Instances: 1994
Number of Attributes: 128
Missing Values? Yes
Area: Social
Date Donated: 2009-07-13
 
Source:
Creator: Michael Redmond (redmond 'at' lasalle.edu); Computer Science; La Salle 
University; Philadelphia, PA, 19141, USA
-- culled from 1990 US Census, 1995 US FBI Uniform Crime Report, 1990 US Law Enforcement Management and Administrative Statistics Survey, available from ICPSR at U of Michigan.
-- Donor: Michael Redmond (redmond 'at' lasalle.edu); Computer Science; La Salle University; Philadelphia, PA, 19141, USA
-- Date: July 2009

Data Set Information:
Many variables are included so that algorithms that select or learn weights for attributes could be tested. However, clearly unrelated attributes were not included; attributes were picked if there was any plausible connection to crime (N=122), plus the attribute to be predicted (Per Capita Violent Crimes). The variables included in the dataset involve the community, such as the percent of the population considered urban, and the median family income, and involving law enforcement, such as per capita number of police officers, and percent of officers assigned to drug units. The per capita violent crimes variable was calculated using population and the sum of crime variables considered violent crimes in the United States: murder, rape, robbery, and assault. There was apparently some controversy in some states concerning the counting of rapes. These resulted in missing values for rape, which resulted in incorrect values for per capita violent crime. These cities are not included in the dataset. Many of these omitted communities were from the midwestern USA. Data is described below based on original values. All numeric data was normalized into the decimal range 0.00-1.00 using an Unsupervised, equal-interval binning method. 
Attributes retain their distribution and skew (hence for example the population 
attribute has a mean value of 0.06 because most communities are small). E.g. An attribute described as 'mean people per household' is actually the normalized (0-1) version of that value. The normalization preserves rough ratios of values WITHIN an attribute (e.g. double the value for double the population within the available precision - except for extreme values (all values more than 3 SD above the mean are normalized to 1.00; all values more than 3 SD below the mean are nromalized to 0.00)).

However, the normalization does not preserve relationships between values BETWEEN attributes (e.g. it would not be meaningful to compare the value for whitePerCap with the value for blackPerCap for a community)
A limitation was that the LEMAS survey was of the police departments with at least 100 officers, plus a random sample of smaller departments. For our purposes, communities not found in both census and crime datasets were omitted. Many communities are missing LEMAS data.
====
Target Variable: ViolentCrimesPerPop (numeric, 98 distinct): ['0.03', '0.04', '0.06', '0.05', '0.02', '0.09', '0.1', '0.07', '0.12', '0.08']
====
Features:

state (numeric, 46 distinct): ['6', '34', '48', '25', '39', '42', '12', '9', '55', '18']
county (numeric, 109 distinct): ['3.0', '17.0', '9.0', '1.0', '7.0', '5.0', '27.0', '21.0', '13.0', '23.0']
community (numeric, 800 distinct): ['11000.0', '51000.0', '21344.0', '77000.0', '52980.0', '79000.0', '30210.0', '57000.0', '63000.0', '16000.0']
fold (numeric, 10 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
population (numeric, 66 distinct): ['0.01', '0.0', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
householdsize (numeric, 93 distinct): ['0.41', '0.39', '0.35', '0.37', '0.33', '0.49', '0.34', '0.45', '0.43', '0.36']
racepctblack (numeric, 100 distinct): ['0.01', '0.02', '0.03', '0.0', '0.04', '0.05', '0.06', '1.0', '0.07', '0.09']
racePctWhite (numeric, 99 distinct): ['0.98', '0.96', '0.97', '0.99', '0.95', '0.93', '0.91', '0.94', '0.92', '0.9']
racePctAsian (numeric, 91 distinct): ['0.02', '0.03', '0.04', '0.05', '0.06', '0.01', '0.09', '0.07', '0.08', '0.11']
racePctHisp (numeric, 91 distinct): ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '1.0', '0.0', '0.07', '0.09']
agePct12t21 (numeric, 93 distinct): ['0.38', '0.41', '0.4', '0.39', '0.36', '0.37', '0.35', '0.34', '0.45', '0.42']
agePct12t29 (numeric, 89 distinct): ['0.49', '0.46', '0.48', '0.5', '0.41', '0.44', '0.42', '0.47', '0.45', '0.52']
agePct16t24 (numeric, 94 distinct): ['0.29', '0.26', '0.28', '0.27', '0.3', '0.31', '0.24', '0.23', '0.32', '0.25']
agePct65up (numeric, 98 distinct): ['0.47', '0.35', '0.33', '0.52', '0.42', '0.45', '0.36', '0.51', '0.48', '0.39']
numbUrban (numeric, 67 distinct): ['0.0', '0.02', '0.03', '0.04', '0.05', '0.06', '0.08', '0.07', '0.09', '0.1']
pctUrban (numeric, 64 distinct): ['1.0', '0.0', '0.99', '0.98', '0.97', '0.91', '0.96', '0.79', '0.86', '0.93']
medIncome (numeric, 99 distinct): ['0.23', '0.16', '0.17', '0.25', '0.22', '0.18', '0.21', '0.15', '0.2', '0.33']
pctWWage (numeric, 96 distinct): ['0.58', '0.53', '0.49', '0.6', '0.57', '0.52', '0.51', '0.64', '0.54', '0.56']
pctWFarmSelf (numeric, 99 distinct): ['0.16', '0.17', '0.19', '0.18', '0.22', '0.21', '0.12', '0.15', '0.14', '0.13']
pctWInvInc (numeric, 96 distinct): ['0.41', '0.46', '0.48', '0.39', '0.38', '0.5', '0.42', '0.52', '0.51', '0.58']
pctWSocSec (numeric, 96 distinct): ['0.56', '0.51', '0.46', '0.53', '0.52', '0.47', '0.49', '0.45', '0.5', '0.58']
pctWPubAsst (numeric, 101 distinct): ['0.1', '0.12', '0.16', '0.21', '0.17', '0.13', '0.09', '0.15', '0.27', '0.14']
pctWRetire (numeric, 93 distinct): ['0.44', '0.4', '0.42', '0.51', '0.45', '0.47', '0.5', '0.62', '0.57', '0.39']
medFamInc (numeric, 98 distinct): ['0.25', '0.21', '0.19', '0.26', '0.22', '0.27', '0.24', '0.28', '0.23', '0.2']
perCapInc (numeric, 98 distinct): ['0.23', '0.22', '0.24', '0.21', '0.25', '0.27', '0.31', '0.26', '0.18', '0.2']
whitePerCap (numeric, 101 distinct): ['0.3', '0.2', '0.28', '0.32', '0.25', '0.29', '0.26', '0.23', '0.22', '0.24']
blackPerCap (numeric, 91 distinct): ['0.18', '0.17', '0.16', '0.19', '0.2', '0.15', '0.21', '0.27', '0.22', '0.23']
indianPerCap (numeric, 86 distinct): ['0.0', '0.16', '0.15', '0.11', '0.13', '0.14', '0.18', '0.17', '0.09', '0.2']
AsianPerCap (numeric, 98 distinct): ['0.18', '0.23', '0.25', '0.26', '0.27', '0.29', '0.22', '0.28', '0.31', '0.21']
OtherPerCap (numeric, 98 distinct): ['0.0', '0.25', '0.2', '0.22', '0.16', '0.26', '0.24', '0.21', '0.27', '0.17']
HispPerCap (numeric, 94 distinct): ['0.3', '0.33', '0.32', '0.26', '0.31', '0.23', '0.28', '0.22', '0.34', '0.39']
NumUnderPov (numeric, 66 distinct): ['0.01', '0.0', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
PctPopUnderPov (numeric, 100 distinct): ['0.08', '0.07', '0.05', '0.1', '0.09', '0.06', '0.12', '0.11', '0.13', '0.19']
PctLess9thGrade (numeric, 97 distinct): ['0.19', '0.11', '0.17', '0.3', '0.18', '0.16', '0.21', '0.27', '0.13', '0.14']
PctNotHSGrad (numeric, 99 distinct): ['0.39', '0.19', '0.15', '0.29', '0.22', '0.4', '0.35', '0.36', '0.24', '0.14']
PctBSorMore (numeric, 96 distinct): ['0.18', '0.21', '0.24', '0.25', '0.23', '0.26', '0.33', '0.34', '0.28', '0.27']
PctUnemployed (numeric, 98 distinct): ['0.24', '0.22', '0.32', '0.27', '0.25', '0.2', '0.35', '0.26', '0.38', '0.28']
PctEmploy (numeric, 96 distinct): ['0.56', '0.58', '0.51', '0.5', '0.52', '0.6', '0.47', '0.46', '0.57', '0.42']
PctEmplManu (numeric, 100 distinct): ['0.26', '0.3', '0.36', '0.39', '0.27', '0.25', '0.35', '0.28', '0.41', '0.24']
PctEmplProfServ (numeric, 96 distinct): ['0.36', '0.39', '0.38', '0.3', '0.43', '0.4', '0.44', '0.32', '0.37', '0.35']
PctOccupManu (numeric, 98 distinct): ['0.32', '0.34', '0.45', '0.39', '0.42', '0.37', '0.4', '0.35', '0.3', '0.33']
PctOccupMgmtProf (numeric, 99 distinct): ['0.36', '0.34', '0.39', '0.31', '0.37', '0.33', '0.35', '0.32', '0.4', '0.42']
MalePctDivorce (numeric, 98 distinct): ['0.56', '0.48', '0.45', '0.55', '0.43', '0.59', '0.42', '0.63', '0.49', '0.38']
MalePctNevMarr (numeric, 96 distinct): ['0.38', '0.3', '0.35', '0.42', '0.36', '0.4', '0.32', '0.39', '0.28', '0.34']
FemalePctDiv (numeric, 91 distinct): ['0.54', '0.5', '0.56', '0.63', '0.61', '0.64', '0.49', '0.59', '0.58', '0.42']
TotalPctDiv (numeric, 94 distinct): ['0.57', '0.54', '0.61', '0.48', '0.64', '0.66', '0.62', '0.42', '0.56', '0.52']
PersPerFam (numeric, 92 distinct): ['0.44', '0.4', '0.49', '0.47', '0.51', '0.42', '0.35', '0.37', '0.58', '0.56']
PctFam2Par (numeric, 101 distinct): ['0.7', '0.64', '0.61', '0.62', '0.54', '0.66', '0.6', '0.8', '0.63', '0.71']
PctKids2Par (numeric, 97 distinct): ['0.72', '0.59', '0.76', '0.68', '0.67', '0.49', '0.62', '0.81', '0.58', '0.82']
PctYoungKids2Par (numeric, 99 distinct): ['0.91', '0.88', '0.86', '0.71', '0.87', '0.84', '0.83', '0.89', '0.7', '0.75']
PctTeen2Par (numeric, 96 distinct): ['0.6', '0.61', '0.63', '0.62', '0.69', '0.64', '0.66', '0.59', '0.71', '0.68']
PctWorkMomYoungKids (numeric, 95 distinct): ['0.51', '0.52', '0.48', '0.55', '0.49', '0.43', '0.5', '0.57', '0.44', '0.4']
PctWorkMom (numeric, 98 distinct): ['0.57', '0.54', '0.49', '0.52', '0.6', '0.51', '0.67', '0.58', '0.59', '0.5']
NumIlleg (numeric, 55 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.09', '0.08']
PctIlleg (numeric, 97 distinct): ['0.09', '0.06', '0.08', '0.04', '0.05', '0.1', '0.13', '0.07', '0.15', '0.03']
NumImmig (numeric, 47 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
PctImmigRecent (numeric, 99 distinct): ['0.0', '0.27', '0.33', '0.29', '0.25', '0.2', '0.31', '0.3', '0.37', '0.18']
PctImmigRec5 (numeric, 100 distinct): ['0.0', '0.42', '0.33', '0.48', '0.32', '0.3', '0.27', '0.17', '0.35', '0.18']
PctImmigRec8 (numeric, 97 distinct): ['0.26', '0.36', '0.39', '0.45', '0.5', '0.46', '0.38', '0.35', '0.3', '0.43']
PctImmigRec10 (numeric, 97 distinct): ['0.43', '0.45', '0.44', '0.47', '0.28', '0.52', '0.41', '0.46', '0.48', '0.51']
PctRecentImmig (numeric, 95 distinct): ['0.01', '0.03', '0.0', '0.02', '0.05', '0.04', '0.07', '0.06', '0.08', '0.11']
PctRecImmig5 (numeric, 97 distinct): ['0.02', '0.01', '0.03', '0.04', '0.06', '0.08', '0.07', '0.0', '0.05', '0.11']
PctRecImmig8 (numeric, 98 distinct): ['0.02', '0.03', '0.01', '0.04', '0.05', '0.08', '0.07', '0.06', '0.09', '0.0']
PctRecImmig10 (numeric, 100 distinct): ['0.02', '0.01', '0.03', '0.04', '0.06', '0.05', '0.07', '0.08', '0.09', '0.1']
PctSpeakEnglOnly (numeric, 98 distinct): ['0.96', '0.95', '0.97', '0.94', '0.92', '0.93', '0.9', '0.89', '0.88', '0.91']
PctNotSpeakEnglWell (numeric, 94 distinct): ['0.03', '0.04', '0.02', '0.05', '0.06', '0.01', '0.08', '0.07', '0.09', '1.0']
PctLargHouseFam (numeric, 99 distinct): ['0.17', '0.2', '0.19', '0.14', '0.18', '0.16', '0.15', '0.12', '0.13', '0.23']
PctLargHouseOccup (numeric, 96 distinct): ['0.19', '0.17', '0.15', '0.21', '0.18', '0.14', '0.16', '0.11', '0.13', '0.12']
PersPerOccupHous (numeric, 96 distinct): ['0.37', '0.36', '0.38', '0.4', '0.34', '0.32', '0.45', '0.29', '0.41', '0.33']
PersPerOwnOccHous (numeric, 94 distinct): ['0.45', '0.47', '0.41', '0.43', '0.4', '0.57', '0.46', '0.51', '0.48', '0.49']
PersPerRentOccHous (numeric, 98 distinct): ['0.32', '0.3', '0.29', '0.36', '0.37', '0.35', '0.28', '0.34', '0.27', '0.26']
PctPersOwnOccup (numeric, 100 distinct): ['0.54', '0.52', '0.57', '0.59', '0.55', '0.56', '0.49', '0.47', '0.48', '0.45']
PctPersDenseHous (numeric, 94 distinct): ['0.06', '0.04', '0.05', '0.03', '0.07', '0.08', '0.1', '0.02', '0.09', '0.12']
PctHousLess3BR (numeric, 100 distinct): ['0.55', '0.53', '0.51', '0.57', '0.58', '0.48', '0.47', '0.46', '0.49', '0.5']
MedNumBR (numeric, 3 distinct): ['0.5', '0.0', '1.0']
HousVacant (numeric, 70 distinct): ['0.01', '0.02', '0.03', '0.04', '0.05', '0.0', '0.06', '0.07', '0.08', '0.09']
PctHousOccup (numeric, 92 distinct): ['0.88', '0.83', '0.84', '0.89', '0.81', '0.87', '0.8', '0.85', '0.82', '0.9']
PctHousOwnOcc (numeric, 99 distinct): ['0.52', '0.47', '0.48', '0.5', '0.55', '0.49', '0.56', '0.59', '0.58', '0.54']
PctVacantBoarded (numeric, 97 distinct): ['0.0', '0.03', '0.05', '0.04', '0.1', '0.06', '0.07', '0.08', '0.09', '0.14']
PctVacMore6Mos (numeric, 98 distinct): ['0.44', '0.37', '0.29', '0.43', '0.47', '0.45', '0.34', '0.46', '0.24', '0.38']
MedYrHousBuilt (numeric, 49 distinct): ['0.0', '0.52', '0.48', '0.54', '0.5', '0.67', '0.56', '0.4', '0.65', '0.63']
PctHousNoPhone (numeric, 99 distinct): ['0.01', '0.03', '0.02', '0.05', '0.04', '0.06', '0.07', '0.1', '0.09', '0.0']
PctWOFullPlumb (numeric, 91 distinct): ['0.0', '0.11', '0.07', '0.13', '0.16', '0.18', '0.09', '0.14', '0.1', '0.2']
OwnOccLowQuart (numeric, 99 distinct): ['0.09', '0.07', '0.06', '0.05', '0.1', '0.13', '0.08', '0.11', '0.16', '0.18']
OwnOccMedVal (numeric, 100 distinct): ['0.08', '0.05', '0.1', '0.09', '0.07', '0.12', '0.06', '0.11', '0.13', '0.04']
OwnOccHiQuart (numeric, 98 distinct): ['0.08', '0.09', '0.07', '0.13', '0.11', '0.12', '0.05', '0.14', '0.06', '0.1']
RentLowQ (numeric, 101 distinct): ['0.13', '0.23', '0.12', '0.1', '0.2', '0.35', '0.16', '0.17', '0.08', '0.14']
RentMedian (numeric, 99 distinct): ['0.19', '0.2', '0.26', '0.14', '0.17', '0.21', '0.15', '0.25', '0.18', '0.16']
RentHighQ (numeric, 99 distinct): ['1.0', '0.26', '0.14', '0.16', '0.24', '0.19', '0.18', '0.23', '0.25', '0.22']
MedRent (numeric, 100 distinct): ['0.17', '0.21', '0.15', '0.23', '0.2', '0.18', '0.29', '0.24', '0.41', '0.22']
MedRentPctHousInc (numeric, 95 distinct): ['0.4', '0.38', '0.37', '0.51', '0.55', '0.44', '0.52', '0.53', '0.49', '0.56']
MedOwnCostPctInc (numeric, 97 distinct): ['0.41', '0.54', '0.51', '0.49', '0.46', '0.37', '0.42', '0.56', '0.63', '0.44']
MedOwnCostPctIncNoMtg (numeric, 70 distinct): ['0.24', '0.22', '0.28', '0.37', '0.31', '0.39', '0.42', '0.32', '0.36', '0.35']
NumInShelters (numeric, 54 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '1.0']
NumStreet (numeric, 53 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.06', '0.05', '0.08', '0.07', '0.09']
PctForeignBorn (numeric, 96 distinct): ['0.03', '0.04', '0.05', '0.02', '0.07', '0.06', '0.1', '0.09', '0.01', '0.08']
PctBornSameState (numeric, 99 distinct): ['0.78', '0.71', '0.79', '0.77', '0.75', '0.48', '0.6', '0.72', '0.8', '0.68']
PctSameHouse85 (numeric, 99 distinct): ['0.59', '0.61', '0.54', '0.53', '0.55', '0.58', '0.6', '0.43', '0.49', '0.67']
PctSameCity85 (numeric, 100 distinct): ['0.74', '0.77', '0.73', '0.67', '0.72', '0.79', '0.8', '0.71', '0.61', '0.63']
PctSameState85 (numeric, 97 distinct): ['0.79', '0.76', '0.81', '0.75', '0.78', '0.82', '0.8', '0.83', '0.74', '0.69']
LemasSwornFT (numeric, 39 distinct): ['0.02', '0.01', '0.03', '0.04', '0.05', '0.07', '0.06', '0.08', '0.13', '0.14']
LemasSwFTPerPop (numeric, 53 distinct): ['0.1', '0.2', '0.15', '0.16', '0.19', '0.17', '0.18', '0.14', '0.13', '0.25']
LemasSwFTFieldOps (numeric, 35 distinct): ['0.98', '0.97', '0.96', '0.95', '0.93', '0.94', '0.92', '0.91', '0.87', '0.85']
LemasSwFTFieldPerPop (numeric, 56 distinct): ['0.19', '0.14', '0.21', '0.2', '0.32', '0.13', '0.17', '0.12', '0.16', '0.25']
LemasTotalReq (numeric, 45 distinct): ['0.02', '0.03', '0.01', '0.04', '0.05', '0.06', '0.07', '0.08', '0.1', '0.09']
LemasTotReqPerPop (numeric, 60 distinct): ['0.14', '0.17', '0.13', '0.12', '0.19', '0.18', '0.15', '0.08', '0.1', '0.11']
PolicReqPerOffic (numeric, 76 distinct): ['0.23', '0.25', '0.24', '0.26', '0.2', '0.3', '0.22', '0.21', '0.37', '0.31']
PolicPerPop (numeric, 53 distinct): ['0.1', '0.2', '0.15', '0.16', '0.19', '0.17', '0.18', '0.14', '0.13', '0.25']
RacialMatchCommPol (numeric, 77 distinct): ['0.78', '0.94', '0.77', '0.62', '0.95', '0.9', '0.84', '0.76', '0.36', '1.0']
PctPolicWhite (numeric, 75 distinct): ['0.72', '0.92', '0.97', '0.79', '0.85', '0.89', '0.88', '0.69', '0.93', '0.99']
PctPolicBlack (numeric, 74 distinct): ['0.0', '0.02', '0.04', '0.06', '0.07', '0.08', '0.03', '0.05', '0.09', '0.12']
PctPolicHisp (numeric, 55 distinct): ['0.0', '0.02', '0.04', '0.03', '0.07', '0.01', '0.16', '0.09', '0.06', '0.05']
PctPolicAsian (numeric, 51 distinct): ['0.0', '1.0', '0.12', '0.05', '0.15', '0.04', '0.06', '0.08', '0.13', '0.07']
PctPolicMinor (numeric, 73 distinct): ['0.07', '0.13', '0.1', '0.01', '0.08', '1.0', '0.3', '0.03', '0.34', '0.09']
OfficAssgnDrugUnits (numeric, 31 distinct): ['0.03', '0.02', '0.04', '0.01', '0.05', '0.07', '0.11', '0.1', '0.08', '0.0']
NumKindsDrugsSeiz (numeric, 16 distinct): ['0.57', '0.5', '0.64', '0.43', '0.36', '0.79', '0.71', '0.93', '0.29', '0.86']
PolicAveOTWorked (numeric, 78 distinct): ['0.19', '0.1', '0.14', '0.17', '0.16', '0.38', '0.11', '0.09', '0.0', '1.0']
LandArea (numeric, 61 distinct): ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.0', '0.09']
PopDens (numeric, 96 distinct): ['0.09', '0.12', '0.14', '0.08', '0.13', '0.06', '0.15', '0.11', '0.16', '0.19']
PctUsePubTrans (numeric, 98 distinct): ['0.01', '0.0', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
PolicCars (numeric, 64 distinct): ['0.02', '0.04', '0.05', '0.06', '0.03', '0.08', '0.09', '0.01', '0.07', '0.11']
PolicOperBudg (numeric, 39 distinct): ['0.02', '0.01', '0.03', '0.04', '0.05', '0.06', '0.07', '0.0', '0.09', '0.08']
LemasPctPolicOnPatr (numeric, 73 distinct): ['0.74', '0.84', '0.79', '0.8', '0.83', '0.81', '0.75', '0.86', '0.78', '0.73']
LemasGangUnitDeploy (numeric, 4 distinct): ['0.0', '0.5', '1.0']
LemasPctOfficDrugUn (numeric, 80 distinct): ['0.0', '1.0', '0.51', '0.36', '0.61', '0.45', '0.43', '0.37', '0.56', '0.52']
PolicBudgPerPop (numeric, 52 distinct): ['0.12', '0.1', '0.14', '0.15', '0.16', '0.13', '0.11', '0.19', '0.09', '0.18']
'''

CONTEXT = "US Census Crime Data per County"
TARGET = CuratedTarget(raw_name="ViolentCrimesPerPop", new_name="Violent Crimes Per Pop",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []