from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: nyc-taxi-green-dec-2016
====
Examples: 581835
====
URL: https://www.openml.org/search?type=data&id=42729
====
Description: String datetime information extracted to numeric columns.Trip Record Data provided by the New York City Taxi and Limousine Commission (TLC) [http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml]. The dataset includes TLC trips of the green line in December 2016. Data was downloaded on 03.11.2018. For a description of all variables in the dataset checkout the TLC homepage [http://www.nyc.gov/html/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf]. The variable 'tip_amount' was chosen as target variable. The variable 'total_amount' is ignored by default, otherwise the target could be predicted deterministically. The date variables 'lpep_pickup_datetime' and 'lpep_dropoff_datetime' (ignored by default) could be used to compute additional time features. In this version, we chose only trips with 'payment_type' == 1 (credit card), as tips are not included for most other payment types. We also removed the variables 'trip_distance' and 'fare_amount' to increase the importance of the categorical features 'PULocationID' and 'DOLocationID'.

https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf

VendorID A code indicating the LPEP provider that provided the record.
1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.

lpep_pickup_datetime The date and time when the meter was engaged.

lpep_dropoff_datetime The date and time when the meter was disengaged.

Passenger_count The number of passengers in the vehicle.
This is a driver-entered value.

Trip_distance The elapsed trip distance in miles reported by the taximeter.

PULocationID TLC Taxi Zone in which the taximeter was engaged

DOLocationID TLC Taxi Zone in which the taximeter was disengaged

RateCodeID The final rate code in effect at the end of the trip.
1= Standard rate
2=JFK
3=Newark
4=Nassau or Westchester
5=Negotiated fare
6=Group ride

Store_and_fwd_flag This flag indicates whether the trip record was held in vehicle
memory before sending to the vendor, aka “store and forward,”
because the vehicle did not have a connection to the server.
Y= store and forward trip
N= not a store and forward trip

Payment_type A numeric code signifying how the passenger paid for the trip.
1= Credit card
2= Cash
3= No charge
4= Dispute
5= Unknown
6= Voided trip

Fare_amount The time-and-distance fare calculated by the meter.

Extra Miscellaneous extras and surcharges. Currently, this only includes
the $0.50 and $1 rush hour and overnight charges

MTA_tax $0.50 MTA tax that is automatically triggered based on the metered
rate in use.

Improvement_surcharge $0.30 improvement surcharge assessed on hailed trips at the flag
drop. The improvement surcharge began being levied in 2015

Tip_amount Tip amount – This field is automatically populated for credit card
tips. Cash tips are not included.

Tolls_amount Total amount of all tolls paid in trip.

Total_amount The total amount charged to passengers. Does not include cash tips.

Trip_type A code indicating whether the trip was a street-hail or a dispatch
that is automatically assigned based on the metered rate in use but
can be altered by the driver.
1= Street-hail
2= Dispatch


====
Target Variable: tip_amount (numeric, 1811 distinct): ['0.0', '1.0', '2.0', '3.0', '1.46', '1.36', '1.56', '1.26', '1.66', '1.76']
====
Features:

VendorID (nominal, 2 distinct): ['2', '1']
store_and_fwd_flag (nominal, 2 distinct): ['N', 'Y']
RatecodeID (nominal, 5 distinct): ['1', '5', '2', '3', '4']
PULocationID (nominal, 233 distinct): ['74', '255', '41', '75', '166', '181', '7', '97', '42', '33']
DOLocationID (nominal, 259 distinct): ['181', '74', '41', '42', '7', '112', '166', '97', '61', '49']
passenger_count (numeric, 10 distinct): ['1', '2', '5', '3', '6', '4', '0', '7', '8', '9']
extra (nominal, 5 distinct): ['0', '0.5', '1', '4.5', '0.22']
mta_tax (nominal, 3 distinct): ['0.5', '0', '-0.5']
tolls_amount (numeric, 105 distinct): ['0.0', '5.54', '2.54', '10.5', '12.5', '11.08', '8.0', '2.08', '16.04', '23.58']
improvement_surcharge (nominal, 3 distinct): ['0.3', '0', '-0.3']
total_amount (numeric, 5377 distinct): ['8.76', '8.16', '9.36', '7.56', '9.96', '7.8', '7.3', '8.8', '8.3', '10.56']
trip_type (nominal, 2 distinct): ['1', '2']
lpep_pickup_datetime_day (numeric, 31 distinct): ['10', '16', '3', '17', '9', '15', '2', '18', '11', '8']
lpep_pickup_datetime_hour (numeric, 24 distinct): ['19', '18', '20', '17', '21', '22', '16', '23', '15', '9']
lpep_pickup_datetime_minute (numeric, 60 distinct): ['10', '39', '34', '45', '33', '38', '35', '29', '9', '44']
lpep_dropoff_datetime_day (numeric, 31 distinct): ['10', '16', '3', '17', '9', '15', '2', '18', '11', '8']
lpep_dropoff_datetime_hour (numeric, 24 distinct): ['19', '18', '20', '21', '17', '22', '23', '16', '15', '0']
lpep_dropoff_datetime_minute (numeric, 60 distinct): ['0', '59', '57', '2', '1', '54', '53', '56', '58', '44']
'''

CONTEXT = "New York City Taxi Green Tip Amount"
TARGET = CuratedTarget(raw_name="tip_amount", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="trip_type", value_mapping={'1': 'Street-hail', '2': 'Dispatch'})]
