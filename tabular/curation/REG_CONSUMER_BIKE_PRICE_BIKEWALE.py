from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: REG_CONSUMER_BIKE_PRICE_BIKEWALE
====
Examples: 9003
====
URL: http://pages.cs.wisc.edu/~anhai/data/784_data/bikes/csv_files/bikewale.csv
====
Description: 

====
Features:

id (int64, 9003 distinct): ['12', '25005', '24976', '24985', '24987', '24991', '24995', '25003', '25006', '24973']
bike_name (object, 437 distinct): ['Bajaj Pulsar 150 DTS- i Standard', 'Bajaj Pulsar 180 DTS- i Standard', 'Bajaj Avenger 220 DTS- i', 'Honda CB Unicorn Standard', 'Yamaha FZ S Standard', 'Bajaj Pulsar NS200 Standard', 'Honda Activa Deluxe', 'Yamaha Fazer Standard', 'Bajaj Pulsar 135 LS Standard', 'Yamaha FZ16 Standard']
city_posted (object, 3 distinct): ['Delhi', 'Bangalore', 'Mumbai']
km_driven (int64, 1157 distinct): ['30000', '40000', '20000', '35000', '25000', '50000', '15000', '10000', '45000', '60000']
color (object, 164 distinct): ['black', 'red', 'blue', 'silver', 'white', 'grey', 'green', 'yellow', 'orange', 'maroon']
fuel_type (object, 1 distinct): ['Petrol']
price (int64, 372 distinct): ['35000', '40000', '25000', '30000', '45000', '50000', '20000', '55000', '60000', '65000']
model_year (int64, 46 distinct): ['2011', '2012', '2010', '2013', '2009', '2007', '2008', '2014', '2006', '2005']
owner_type (object, 5 distinct): ['First', 'Second', 'Third', 'Fourth', 'Fifth']
url (object, 9003 distinct): ['http://www.bikewale.com/used/bikes-in-mumbai/tvs-apachertr160-S12/', 'http://www.bikewale.com/used/bikes-in-newdelhi/bajaj-pulsar180dtsi-S25005/', 'http://www.bikewale.com/used/bikes-in-newdelhi/bajaj-pulsar150dtsi-S24976/', 'http://www.bikewale.com/used/bikes-in-mumbai/honda-activa-S24985/', 'http://www.bikewale.com/used/bikes-in-newdelhi/royalenfield-bulletelectratwinspark-S24987/', 'http://www.bikewale.com/used/bikes-in-bangalore/yamaha-ss125-S24991/', 'http://www.bikewale.com/used/bikes-in-bangalore/royalenfield-classic350-S24995/', 'http://www.bikewale.com/used/bikes-in-mumbai/honda-cbunicorn-S25003/', 'http://www.bikewale.com/used/bikes-in-newdelhi/hero-maestro-S25006/', 'http://www.bikewale.com/used/bikes-in-mumbai/royalenfield-thunderbird350-S24973/']
'''

CONTEXT = "Bike and scooters from bikewale website in India"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id", "url"]
FEATURES = []

DESCRIPTION = '''
'''