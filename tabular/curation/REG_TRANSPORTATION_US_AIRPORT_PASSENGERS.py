from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: USA-Airport-Dataset
====
Examples: 3606803
====
URL: https://www.openml.org/search?type=data&id=43479
====
Description: What is it ?
This dataset is a record of 3.5 Million+ US Domestic Flights  from 1990 to 2009. It has been taken from OpenFlights website which have a huge database of different travelling mediums across the globe. I came across this dataset while I was preparing for a hackathon and thought it should on kaggle's dataset list.
What's in it ?
Here is some info about the attributes present in the dataset:

Origin_airport: Three letter airport code of the origin airport
Destination_airport: Three letter airport code of the destination airport
Origin_city: Origin city name
Destination_city: Destination city name
Passengers: Number of passengers transported from origin to destination
Seats: Number of seats available on flights from origin to destination
Flights: Number of flights between origin and destination (multiple records for one month, many with flights  1)
Distance:  Distance (to nearest mile) flown between origin and destination
Fly_date: The date (yyyymm) of flight
Origin_population: Origin city's population as reported by US Census
Destination_population: Destination city's population as reported by US Census

Where did you get it ?
I would like to thank the original author of this dataset Jacob Perkins for putting such a good effort and collecting this data. 

I will be updating this dataset for attributes that might reveal some more information about this dataset.
Thanks for stopping by.
Peace.
====
Features:

Origin_airport (string, 683 distinct): ['ORD', 'ATL', 'DFW', 'DTW', 'MSP', 'LAX', 'CLT', 'PHL', 'IAH', 'EWR']
Destination_airport (string, 708 distinct): ['ORD', 'ATL', 'DFW', 'DTW', 'MSP', 'CLT', 'LAX', 'IAH', 'PHL', 'EWR']
Origin_city (string, 535 distinct): ['Chicago, IL', 'Atlanta, GA', 'Dallas, TX', 'Detroit, MI', 'New York, NY', 'Houston, TX', 'Minneapolis, MN', 'Washington, DC', 'Los Angeles, CA', 'Charlotte, NC']
Destination_city (string, 548 distinct): ['Chicago, IL', 'Atlanta, GA', 'Dallas, TX', 'New York, NY', 'Detroit, MI', 'Houston, TX', 'Minneapolis, MN', 'Washington, DC', 'Charlotte, NC', 'Los Angeles, CA']
Passengers (numeric, 37484 distinct): ['0', '50', '37', '49', '122', '47', '48', '44', '45', '43']
Seats (numeric, 41098 distinct): ['0', '50', '122', '150', '100', '124', '142', '137', '148', '128']
Flights (numeric, 920 distinct): ['1', '2', '30', '31', '4', '3', '29', '5', '28', '27']
Distance (numeric, 2776 distinct): ['224', '296', '337', '256', '370', '487', '328', '528', '228', '229']
Fly_date (string, 240 distinct): ['2007-12-01', '2004-12-01', '2007-06-01', '2005-01-01', '2006-12-01', '2004-10-01', '2002-12-01', '2008-06-01', '2004-09-01', '2008-02-01']
Origin_population (numeric, 6679 distinct): ['19031272', '18903872', '18664180', '19161134', '18724160', '18797710', '18572324', '18385002', '18235464', '18490270']
Destination_population (numeric, 6696 distinct): ['19031272', '18903872', '19161134', '18664180', '18797710', '18724160', '18572324', '18385002', '18235464', '16720340']
Org_airport_lat (numeric, 478 distinct): ['41.9786', '33.6367', '32.8968', '42.2124', '44.882', '33.9425', '35.214', '39.8719', '29.9844', '40.6925']
Org_airport_long (numeric, 478 distinct): ['-87.9048', '-84.4281', '-97.038', '-83.3534', '-93.2218', '-118.408', '-80.9431', '-75.2411', '-95.3414', '-74.1687']
Dest_airport_lat (numeric, 485 distinct): ['41.9786', '33.6367', '32.8968', '42.2124', '44.882', '35.214', '33.9425', '29.9844', '39.8719', '40.6925']
Dest_airport_long (numeric, 485 distinct): ['-87.9048', '-84.4281', '-97.038', '-83.3534', '-93.2218', '-80.9431', '-118.408', '-95.3414', '-75.2411', '-74.1687']
'''

CONTEXT = "US Domestic Flights"
TARGET = CuratedTarget(raw_name="Passengers", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Fly_date", feat_type=FeatureType.DATE)]
