from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: sobhanmoosavi/us-accidents/US_Accidents_March23.csv
====
Examples: 7728394
====
URL: https://www.kaggle.com/sobhanmoosavi/us-accidents/US_Accidents_March23.csv
====
Description: 
US Accidents (2016 - 2023)
A Countrywide Traffic Accident Dataset (2016 - 2023)

About Dataset
Description
This is a countrywide car accident dataset that covers 49 states of the USA. The accident data were collected from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks. The dataset currently contains approximately 7.7 million accident records. For more information about this dataset, please visit here.

Acknowledgements
If you use this dataset, please kindly cite the following papers:

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

Content
This dataset was collected in real-time using multiple Traffic APIs. It contains accident data collected from February 2016 to March 2023 for the Contiguous United States. For more details about this dataset, please visit [here].

Inspiration
The US-Accidents dataset can be used for numerous applications, such as real-time car accident prediction, studying car accident hotspot locations, casualty analysis, extracting cause and effect rules to predict car accidents, and studying the impact of precipitation or other environmental stimuli on accident occurrence. The most recent release of the dataset can also be useful for studying the impact of COVID-19 on traffic behavior and accidents.

Sampled Data (New!)
For those requiring a smaller, more manageable dataset, a sampled version is available which includes 500,000 accidents. This sample is extracted from the original dataset for easier handling and analysis.

Other Details
Please note that the dataset may be missing data for certain days, which could be due to network connectivity issues during data collection. Regrettably, the dataset will no longer be updated, and this version should be considered the latest.

Usage Policy and Legal Disclaimer
This dataset is being distributed solely for research purposes under the Creative Commons Attribution-Noncommercial-ShareAlike license (CC BY-NC-SA 4.0). By downloading the dataset, you agree to use it only for non-commercial, research, or academic applications. If you use this dataset, it is necessary to cite the papers mentioned above.

Inquiries or need help?
For any inquiries or assistance, please contact Sobhan Moosavi at sobhan.mehr84@gmail.com

====
Features:

ID (object, 7728394 distinct): ['A-1', 'A-5191324', 'A-5191336', 'A-5191335', 'A-5191334', 'A-5191333', 'A-5191332', 'A-5191331', 'A-5191330', 'A-5191329']
Source (object, 3 distinct): ['Source1', 'Source2', 'Source3']
Severity (int64, 4 distinct): ['2', '3', '4', '1']
Start_Time (object, 6131796 distinct): ['2021-01-26 16:16:13', '2021-01-26 16:17:33', '2021-02-16 06:42:43', '2021-11-21 18:37:51', '2020-12-16 13:53:25', '2021-02-16 06:43:35', '2021-05-03 06:29:42', '2017-05-15 09:22:55', '2020-09-30 12:41:30', '2021-04-26 08:58:47']
End_Time (object, 6705355 distinct): ['2021-11-22 08:00:00', '2017-05-15 15:22:55', '2019-10-26 09:14:51', '2020-02-14 00:00:00', '2018-11-25 02:51:02', '2020-02-12 00:00:00', '2020-01-25 00:00:00', '2020-02-15 00:00:00', '2021-07-12 23:41:50', '2020-02-07 00:00:00']
Start_Lat (float64, 2437160 distinct): ['37.8085', '33.9414', '34.8588', '42.4765', '33.745', '34.8589', '40.8479', '34.0394', '33.8763', '25.7891']
Start_Lng (float64, 2482533 distinct): ['-122.3669', '-118.0966', '-82.2604', '-84.3903', '-83.1118', '-73.9428', '-80.1659', '-82.2599', '-118.3683', '-80.2101']
End_Lat (float64, 1568172 distinct): ['28.45', '25.7018', '25.6843', '28.4499', '25.6863', '25.8894', '25.9248', '25.7332', '28.4502', '28.4214']
End_Lng (float64, 1605789 distinct): ['-81.4714', '-80.3342', '-80.4166', '-81.4772', '-80.4165', '-80.2933', '-80.3366', '-81.3998', '-78.6802', '-81.4777']
Distance(mi) (float64, 22382 distinct): ['0.0', '0.01', '0.008', '0.009', '0.01', '0.007', '0.011', '0.03', '0.024', '0.028']
Description (object, 3761578 distinct): ['A crash has occurred causing no to minimum delays. Use caution.', 'Accident', 'An unconfirmed report of a crash has been received. Use caution.', 'A crash has occurred use caution.', 'A crash has occurred with minimal delay to traffic. Prepare to slow or move over for worker safety.', 'A disabled vehicle is creating a hazard causing no to minimum delays. Use caution.', 'At I-15 - Accident.', 'At I-5 - Accident.', 'Incident on I-95 SB near I-95 Drive with caution.', 'Incident on I-95 NB near I-95 Drive with caution.']
Street (object, 336306 distinct): ['I-95 N', 'I-95 S', 'I-5 N', 'I-10 E', 'I-10 W', 'I-5 S', 'I-80 W', 'I-80 E', 'I-405 N', 'I-75 N']
City (object, 13678 distinct): ['Miami', 'Houston', 'Los Angeles', 'Charlotte', 'Dallas', 'Orlando', 'Austin', 'Raleigh', 'Nashville', 'Baton Rouge']
County (object, 1871 distinct): ['Los Angeles', 'Miami-Dade', 'Orange', 'Harris', 'Dallas', 'Mecklenburg', 'Montgomery', 'Wake', 'San Bernardino', 'Travis']
State (object, 49 distinct): ['CA', 'FL', 'TX', 'SC', 'NY', 'NC', 'VA', 'PA', 'MN', 'OR']
Zipcode (object, 825094 distinct): ['91761', '91706', '92407', '92507', '33186', '32819', '91765', '33169', '90023', '92324']
Country (object, 1 distinct): ['US']
Timezone (object, 4 distinct): ['US/Eastern', 'US/Pacific', 'US/Central', 'US/Mountain']
Airport_Code (object, 2045 distinct): ['KCQT', 'KRDU', 'KMCJ', 'KBNA', 'KCLT', 'KORL', 'KMIA', 'KBTR', 'KOPF', 'KDAL']
Weather_Timestamp (object, 941331 distinct): ['2022-03-13 01:53:00', '2021-01-26 15:53:00', '2022-05-13 16:53:00', '2022-03-13 01:55:00', '2021-01-15 22:53:00', '2022-05-17 15:53:00', '2022-05-13 15:53:00', '2022-04-29 14:53:00', '2022-04-22 16:53:00', '2022-04-13 16:53:00']
Temperature(F) (float64, 860 distinct): ['77.0', '73.0', '68.0', '72.0', '75.0', '70.0', '63.0', '59.0', '64.0', '79.0']
Wind_Chill(F) (float64, 1001 distinct): ['73.0', '72.0', '75.0', '77.0', '70.0', '63.0', '79.0', '68.0', '64.0', '66.0']
Humidity(%) (float64, 100 distinct): ['93.0', '100.0', '87.0', '90.0', '89.0', '96.0', '84.0', '81.0', '82.0', '86.0']
Pressure(in) (float64, 1144 distinct): ['29.96', '29.99', '30.01', '29.94', '30.04', '29.97', '30.03', '29.91', '30.0', '29.95']
Visibility(mi) (float64, 92 distinct): ['10.0', '7.0', '9.0', '8.0', '5.0', '6.0', '2.0', '4.0', '3.0', '1.0']
Wind_Direction (object, 24 distinct): ['CALM', 'S', 'SSW', 'W', 'WNW', 'NW', 'Calm', 'SW', 'WSW', 'SSE']
Wind_Speed(mph) (float64, 184 distinct): ['0.0', '5.0', '6.0', '3.0', '7.0', '8.0', '9.0', '10.0', '12.0', '4.6']
Precipitation(in) (float64, 299 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
Weather_Condition (object, 144 distinct): ['Fair', 'Mostly Cloudy', 'Cloudy', 'Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Scattered Clouds', 'Light Snow', 'Fog']
Amenity (bool, 2 distinct): ['0', '1']
Bump (bool, 2 distinct): ['0', '1']
Crossing (bool, 2 distinct): ['0', '1']
Give_Way (bool, 2 distinct): ['0', '1']
Junction (bool, 2 distinct): ['0', '1']
No_Exit (bool, 2 distinct): ['0', '1']
Railway (bool, 2 distinct): ['0', '1']
Roundabout (bool, 2 distinct): ['0', '1']
Station (bool, 2 distinct): ['0', '1']
Stop (bool, 2 distinct): ['0', '1']
Traffic_Calming (bool, 2 distinct): ['0', '1']
Traffic_Signal (bool, 2 distinct): ['0', '1']
Turning_Loop (bool, 1 distinct): ['0']
Sunrise_Sunset (object, 2 distinct): ['Day', 'Night']
Civil_Twilight (object, 2 distinct): ['Day', 'Night']
Nautical_Twilight (object, 2 distinct): ['Day', 'Night']
Astronomical_Twilight (object, 2 distinct): ['Day', 'Night']
'''

CONTEXT = "US Accidents between 2016 and 2023"
TARGET = CuratedTarget(raw_name="Severity", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["ID"]
FEATURES = [CuratedFeature(raw_name="Weather_Timestamp", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Start_Time", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="End_Time", feat_type=FeatureType.DATE),]

DESCRIPTION = '''
US Accidents (2016 - 2023)
A Countrywide Traffic Accident Dataset (2016 - 2023)

About Dataset
Description
This is a countrywide car accident dataset that covers 49 states of the USA. The accident data were collected from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks. The dataset currently contains approximately 7.7 million accident records. For more information about this dataset, please visit here.

Acknowledgements
If you use this dataset, please kindly cite the following papers:

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

Content
This dataset was collected in real-time using multiple Traffic APIs. It contains accident data collected from February 2016 to March 2023 for the Contiguous United States. For more details about this dataset, please visit [here].

Inspiration
The US-Accidents dataset can be used for numerous applications, such as real-time car accident prediction, studying car accident hotspot locations, casualty analysis, extracting cause and effect rules to predict car accidents, and studying the impact of precipitation or other environmental stimuli on accident occurrence. The most recent release of the dataset can also be useful for studying the impact of COVID-19 on traffic behavior and accidents.

Sampled Data (New!)
For those requiring a smaller, more manageable dataset, a sampled version is available which includes 500,000 accidents. This sample is extracted from the original dataset for easier handling and analysis.

Other Details
Please note that the dataset may be missing data for certain days, which could be due to network connectivity issues during data collection. Regrettably, the dataset will no longer be updated, and this version should be considered the latest.

Usage Policy and Legal Disclaimer
This dataset is being distributed solely for research purposes under the Creative Commons Attribution-Noncommercial-ShareAlike license (CC BY-NC-SA 4.0). By downloading the dataset, you agree to use it only for non-commercial, research, or academic applications. If you use this dataset, it is necessary to cite the papers mentioned above.

Inquiries or need help?
For any inquiries or assistance, please contact Sobhan Moosavi at sobhan.mehr84@gmail.com
'''