from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: Myanmar-Air-Quality(2019-to-2020-Oct)
====
Examples: 5122
====
URL: https://www.openml.org/search?type=data&id=43748
====
Description: Context
Since Myanmar is one of the developing countries, a lot of factories were set up and the number of cars increased speedily during the previous years. Therefore, Myanmar's air quality was also dramatically decreasing during the last years. Moreover,  Myanmar air quality reached no.4 in the worst air quality globally in 2019. So, I created this dataset to analyze and to try some predictions.
Content
Data is from Purple.com and cleaned by using PowerBI. 
Acknowledgements
This dataset is a part of the project which is initialized to compete Myanmar's air quality visualization competitions. So, I would like to give credits to my friends who participated in that competition with me. 
Inspiration
I hope this dataset can help the field of data science and the air quality of Myanmar. Context
Since Myanmar is one of the developing countries, a lot of factories were set up and the number of cars increased speedily during the previous years. Therefore, Myanmar's air quality was also dramatically decreasing during the last years. Moreover,  Myanmar air quality reached no.4 in the worst air quality globally in 2019. So, I created this dataset to analyze and to try some predictions.
Content
Data is from PurpleAir.com and cleaned by using PowerBI. 
Acknowledgements
This dataset is a part of the project which is initialized to compete Myanmar's air quality visualization competitions. So, I would like to give credits to my friends who participated in that competition with me. 
Inspiration
I hope this dataset can help the field of data science and the air quality of Myanmar. Context
Since Myanmar is one of the developing countries, a lot of factories were set up and the number of cars increased speedily during the previous years. Therefore, Myanmar's air quality was also dramatically decreasing during the last years. Moreover,  Myanmar air quality reached no.4 in the worst air quality globally in 2019. So, I created this dataset to analyze and to try some predictions.
Content
Data is from PurpleAir.com and cleaned by using PowerBI. 
Acknowledgements
This dataset is a part of the project which is initialized to compete Myanmar's air quality visualization competitions. So, I would like to give credits to my friends who participated in that competition with me. 
Inspiration
I hope this dataset can help the field of data science and the air quality of Myanmar.
====
Features:

City (string, 2 distinct): ['Yangon', 'Mandalay']
Center (string, 14 distinct): ['7 Miles Mayangone', 'Ahlone Myanmar Center for Responsible Business', 'American Center', 'Beca Myanmar (Outside)', 'GEMS Condo', 'Pun Hlaing Dulwich College', 'Star City Dulwich College', 'Thin Gan Gyun Yangon International School (Outside)', 'UNOPS Myanmar', 'Yangon-HO']
Date (string, 378 distinct): ['4/26/2020 0:00', '6/25/2020 0:00', '7/4/2020 0:00', '7/3/2020 0:00', '7/2/2020 0:00', '7/1/2020 0:00', '6/30/2020 0:00', '6/29/2020 0:00', '6/28/2020 0:00', '6/27/2020 0:00']
Year (numeric, 2 distinct): ['2020', '2019']
Month (string, 12 distinct): ['October', 'May', 'July', 'August', 'June', 'September', 'April', 'December', 'January', 'March']
Season (string, 3 distinct): ['Rainy Season', 'Hot Season', 'Cool Season']
PM1_0 (numeric, 2430 distinct): ['19.18', '0.0', '9.71', '4.3', '6.48', '4.35', '4.61', '7.45', '5.03', '3.92']
PM2_5 (numeric, 2736 distinct): ['28.92', '0.0', '7.87', '5.53', '5.95', '11.58', '11.74', '6.77', '9.28', '5.24']
PM10 (numeric, 2762 distinct): ['33.4', '0.0', '9.72', '20.2', '9.81', '7.33', '8.86', '9.25', '7.7', '4.59']
Temperature_F (numeric, 1577 distinct): ['91.19', '88.81', '92.87', '90.63', '90.64', '92.43', '92.62', '90.4', '89.08', '89.17']
Humidity_% (numeric, 2586 distinct): ['53.36', '51.0', '49.37', '55.58', '44.37', '50.3', '52.24', '46.52', '44.26', '46.56']
AQI (numeric, 2686 distinct): ['86.37', '0.0', '32.79', '48.92', '28.21', '29.79', '21.83', '23.04', '52.54', '24.79']
New_cases (numeric, 87 distinct): ['0', '1', '2', '4', '3', '5', '13', '6', '26', '8']
Cumulative_cases (numeric, 168 distinct): ['0', '353', '206', '341', '224', '359', '201', '360', '286', '261']
New_deaths (numeric, 30 distinct): ['0', '1', '25', '27', '32', '39', '28', '33', '2', '31']
Cumulative_deaths (numeric, 59 distinct): ['0', '6', '5', '1', '3', '4', '8', '14', '7', '664']
'''

CONTEXT = "Myanmar Air Quality Assessment"
TARGET = CuratedTarget(raw_name="New_cases", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Cumulative_cases", "Cumulative_deaths", "New_deaths", "Year", "Month"]
FEATURES = [CuratedFeature(raw_name="Date", feat_type=FeatureType.DATE)]