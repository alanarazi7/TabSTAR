from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask
from tabular.preprocessing.textual import convert_currency_k_m, convert_weight_lbs, remove_k

'''
Dataset Name: FIFA20-Players-Dataset-with-Stats--Images
====
Examples: 14999
====
URL: https://www.openml.org/search?type=data&id=43766
====
Description: About Dataset
This data set includes15K Fifa20 Players with 15+ features and their images , including their position, age, and Country, and many more. It can be used for learning Statistics, Performing Data Analysis, and Data Visualization using various libraries like Seaborn, Pandas-Bokeh, and Plotly. It can be used to plot various Plots to understand the comparison between various features.
References

Sofifa.com
====
Features:

Name (string, 14878 distinct): ['Liam Kelly', 'Adama Traor', 'Danny Rose', 'Jamaal Lascelles', 'Rodrigo Battaglia', 'Richard Snchez', 'Mbaye Diagne', 'Hans Hateboer', 'Tom Davies', 'Baptiste Santamaria']
Country (string, 161 distinct): ['England', 'Spain', 'Germany', 'France', 'Brazil', 'Argentina', 'Italy', 'Colombia', 'Netherlands', 'Japan']
Position (string, 615 distinct): ['CB', 'GK', 'ST', 'CDM,CM', 'CM,CDM', 'LB', 'RB', 'CM', 'CDM', 'CB,RB']
Age (numeric, 27 distinct): ['27', '25', '26', '23', '24', '22', '28', '29', '21', '30']
Overall (numeric, 35 distinct): ['65', '64', '66', '67', '68', '69', '63', '70', '62', '72']
Potential (numeric, 35 distinct): ['65', '64', '66', '67', '68', '69', '63', '70', '62', '72']
Club (string, 667 distinct): ['RC Celta', 'Borussia Mnchengladbach', 'FC Barcelona', 'Real Valladolid CF', 'Atltico Madrid', 'Real Madrid', 'Manchester United', 'Hertha BSC', 'Everton', 'AS Monaco']
ID (numeric, 14944 distinct): ['213135', '215998', '216054', '202642', '227813', '218339', '219576', '219792', '220093', '220131']
Height (string, 21 distinct): ['6\'0"', '5\'10"', '5\'9"', '5\'11"', '6\'2"', '6\'1"', '6\'3"', '5\'8"', '5\'7"', '6\'4"']
Weight (string, 55 distinct): ['165lbs', '154lbs', '176lbs', '172lbs', '159lbs', '161lbs', '163lbs', '170lbs', '168lbs', '174lbs']
Foot (string, 2 distinct): ['Right', 'Left']
BOV (numeric, 36 distinct): ['65', '66', '69', '67', '68', '70', '64', '71', '63', '72']
BP (string, 15 distinct): ['CB', 'ST', 'CAM', 'GK', 'CDM', 'CM', 'RW', 'RM', 'RB', 'LB']
Growth (numeric, 24 distinct): ['0', '1', '3', '5', '7', '6', '4', '2', '9', '8']
Joined (string, 1655 distinct): ['Jul 1, 2019', 'Jul 1, 2018', 'Jul 1, 2017', 'Jul 1, 2016', 'Jul 1, 2015', 'Jan 1, 2018', 'Jul 1, 2014', 'Jan 1, 2019', 'Jul 3, 2019', 'Jul 8, 2019']
Loan_Date_End (string, 24 distinct): ['Jun 30, 2020', 'Dec 31, 2019', 'May 31, 2020', 'Jan 1, 2020', 'Jun 30, 2021', 'Jan 31, 2020', 'Nov 30, 2019', 'Aug 27, 2020', 'Nov 22, 2020', 'Dec 1, 2019']
Value (string, 214 distinct): ['1.1M', '525K', '1.2M', '475K', '450K', '675K', '1M', '400K', '650K', '550K']
Wage (string, 141 distinct): ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K']
Release_Clause (string, 997 distinct): ['0', '1.1M', '1.3M', '1.2M', '1.4M', '1.5M', '1.6M', '1.7M', '1M', '1.8M']
Attacking (numeric, 347 distinct): ['284', '279', '281', '294', '283', '272', '278', '264', '277', '266']
Crossing (numeric, 87 distinct): ['65', '62', '64', '60', '58', '61', '63', '59', '68', '66']
Finishing (numeric, 93 distinct): ['65', '64', '58', '62', '63', '60', '66', '61', '59', '67']
Heading_Accuracy (numeric, 89 distinct): ['58', '65', '60', '62', '64', '59', '55', '68', '61', '66']
Short_Passing (numeric, 83 distinct): ['65', '64', '68', '63', '67', '66', '62', '70', '69', '60']
Volleys (numeric, 86 distinct): ['59', '58', '55', '52', '49', '56', '53', '48', '57', '60']
Skill (numeric, 385 distinct): ['311', '295', '302', '306', '291', '309', '284', '313', '299', '300']
Dribbling (numeric, 92 distinct): ['65', '66', '64', '63', '68', '67', '62', '70', '69', '71']
Curve (numeric, 89 distinct): ['58', '60', '64', '65', '63', '62', '59', '55', '68', '57']
FK_Accuracy (numeric, 89 distinct): ['40', '42', '39', '35', '32', '45', '38', '41', '55', '43']
Long_Passing (numeric, 85 distinct): ['62', '65', '63', '59', '64', '60', '58', '66', '61', '68']
Ball_Control (numeric, 90 distinct): ['65', '64', '66', '68', '67', '63', '70', '62', '69', '72']
Movement (numeric, 325 distinct): ['337', '350', '349', '345', '353', '343', '331', '342', '324', '344']
Acceleration (numeric, 86 distinct): ['68', '67', '74', '69', '75', '76', '66', '71', '77', '72']
Sprint_Speed (numeric, 84 distinct): ['68', '69', '72', '67', '73', '75', '76', '66', '74', '77']
Agility (numeric, 80 distinct): ['72', '68', '70', '67', '74', '73', '71', '69', '75', '66']
Reactions (numeric, 59 distinct): ['65', '64', '63', '62', '66', '60', '58', '68', '61', '67']
Balance (numeric, 81 distinct): ['68', '70', '72', '71', '67', '66', '69', '73', '74', '64']
Power (numeric, 279 distinct): ['339', '333', '324', '331', '308', '317', '315', '323', '330', '326']
Shot_Power (numeric, 79 distinct): ['68', '70', '65', '66', '64', '62', '67', '63', '72', '59']
Jumping (numeric, 71 distinct): ['72', '70', '71', '73', '68', '67', '74', '69', '66', '65']
Stamina (numeric, 85 distinct): ['68', '70', '72', '71', '73', '74', '66', '69', '67', '75']
Strength (numeric, 75 distinct): ['68', '73', '69', '70', '72', '71', '67', '75', '74', '65']
Long_Shots (numeric, 90 distinct): ['59', '58', '62', '65', '60', '64', '63', '66', '68', '61']
Mentality (numeric, 335 distinct): ['288', '271', '275', '293', '266', '286', '282', '279', '302', '280']
Aggression (numeric, 86 distinct): ['68', '70', '65', '72', '67', '66', '60', '58', '69', '73']
Interceptions (numeric, 87 distinct): ['62', '64', '65', '63', '66', '67', '61', '60', '68', '70']
Positioning (numeric, 94 distinct): ['65', '62', '64', '58', '68', '66', '60', '63', '67', '59']
Vision (numeric, 86 distinct): ['65', '59', '58', '64', '60', '62', '63', '68', '57', '66']
Penalties (numeric, 86 distinct): ['55', '59', '60', '58', '48', '49', '62', '61', '52', '45']
Composure (numeric, 85 distinct): ['65', '62', '60', '64', '58', '63', '68', '59', '66', '55']
Defending (numeric, 244 distinct): ['194', '189', '193', '191', '195', '184', '192', '196', '188', '186']
Marking (numeric, 91 distinct): ['65', '62', '64', '60', '66', '63', '68', '58', '67', '61']
Standing_Tackle (numeric, 88 distinct): ['64', '65', '66', '68', '67', '63', '70', '62', '69', '72']
Sliding_Tackle (numeric, 86 distinct): ['64', '62', '63', '65', '66', '68', '61', '67', '60', '59']
Goalkeeping (numeric, 219 distinct): ['53', '52', '51', '54', '56', '50', '49', '55', '57', '48']
GK_Diving (numeric, 62 distinct): ['8', '7', '10', '12', '13', '14', '9', '11', '6', '15']
GK_Handling (numeric, 63 distinct): ['10', '8', '11', '12', '14', '9', '7', '13', '6', '15']
GK_Kicking (numeric, 77 distinct): ['9', '12', '7', '10', '13', '8', '14', '11', '6', '15']
GK_Positioning (numeric, 64 distinct): ['8', '10', '9', '11', '7', '12', '14', '13', '15', '6']
GK_Reflexes (numeric, 66 distinct): ['11', '8', '9', '10', '13', '7', '12', '14', '6', '15']
Total_Stats (numeric, 1297 distinct): ['1738', '1762', '1694', '1721', '1731', '1637', '1662', '1772', '1679', '1658']
Base_Stats (numeric, 225 distinct): ['357', '364', '361', '370', '375', '362', '371', '355', '377', '367']
W/F (numeric, 5 distinct): ['3', '2', '4', '5', '1']
SM (numeric, 5 distinct): ['3', '2', '1', '4', '5']
A/W (string, 3 distinct): ['Medium', 'High', 'Low']
D/W (string, 3 distinct): ['Medium', 'High', 'Low']
IR (numeric, 5 distinct): ['1', '2', '3', '4', '5']
PAC (numeric, 69 distinct): ['67', '69', '71', '66', '68', '70', '74', '76', '73', '64']
SHO (numeric, 79 distinct): ['62', '60', '63', '64', '61', '65', '59', '67', '66', '58']
PAS (numeric, 66 distinct): ['61', '62', '60', '59', '63', '64', '65', '58', '66', '57']
DRI (numeric, 68 distinct): ['65', '64', '68', '67', '66', '69', '63', '70', '71', '62']
DEF (numeric, 78 distinct): ['63', '62', '64', '65', '66', '61', '60', '67', '68', '59']
PHY (numeric, 60 distinct): ['67', '69', '66', '70', '72', '68', '71', '65', '74', '64']
Hits (string, 505 distinct): ['2', '3', '1', '4', '5', '6', '7', '8', '9', '10']
'''

CONTEXT = "FIFA 2020 Players Stats Value Prediction"
TARGET = CuratedTarget(raw_name="Value", new_name="Market Value in Millions", task_type=SupervisedTask.REGRESSION,
                       processing_func=convert_currency_k_m)
COLS_TO_DROP = ["ID"]
FEATURES = [CuratedFeature(raw_name="Wage", new_name="Wage in Millions", processing_func=convert_currency_k_m,
                           feat_type=FeatureType.NUMERIC),
            CuratedFeature(raw_name="Release_Clause", new_name="Release Clause in Millions",
                           processing_func=convert_currency_k_m, feat_type=FeatureType.NUMERIC),
            CuratedFeature(raw_name="Weight", new_name="Weight in LBS", processing_func=convert_weight_lbs,
                           feat_type=FeatureType.NUMERIC),
            CuratedFeature(raw_name="Joined", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Loan_Date_End", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Hits", processing_func=remove_k, feat_type=FeatureType.NUMERIC),]