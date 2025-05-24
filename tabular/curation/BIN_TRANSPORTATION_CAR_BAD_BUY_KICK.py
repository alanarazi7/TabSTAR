from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import FeatureType, SupervisedTask

'''
Dataset Name: kick
====
Examples: 72983
====
URL: https://www.openml.org/search?type=data&id=41162
====
Description: One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that the vehicle might have serious issues that prevent it from being sold to customers. The auto community calls these unfortunate purchases &quot;kicks&quot;.

Kicked cars often result when there are tampered odometers, mechanical issues the dealer is not able to address, issues with getting the vehicle title from the seller, or some other unforeseen problem. Kick cars can be very costly to dealers after transportation cost, throw-away repair work, and market losses in reselling the vehicle.

Modelers who can figure out which cars have a higher risk of being kick can provide real value to dealerships trying to provide the best inventory selection possible to their customers.

The challenge of this competition is to predict if the car purchased at the Auction is a Kick (bad buy).
====
Target Variable: IsBadBuy (nominal, 2 distinct): ['0', '1']
====
Features:

PurchDate (numeric, 517 distinct): ['1290470400.0', '1235520000.0', '1291766400.0', '1286928000.0', '1251244800.0', '1288742400.0', '1234915200.0', '1289952000.0', '1264032000.0', '1287532800.0']
Auction (nominal, 3 distinct): ['MANHEIM', 'OTHER', 'ADESA']
VehYear (numeric, 10 distinct): ['2006.0', '2005.0', '2007.0', '2004.0', '2008.0', '2003.0', '2002.0', '2001.0', '2009.0', '2010.0']
VehicleAge (numeric, 10 distinct): ['4', '3', '5', '2', '6', '7', '1', '8', '9', '0']
Make (nominal, 33 distinct): ['CHEVROLET', 'DODGE', 'FORD', 'CHRYSLER', 'PONTIAC', 'KIA', 'SATURN', 'NISSAN', 'HYUNDAI', 'JEEP']
Model (nominal, 1063 distinct): ['PT CRUISER', 'IMPALA', 'TAURUS', 'CALIBER', 'CARAVAN GRAND FWD V6', 'MALIBU 4C', 'TAURUS 3.0L V6 EFI', 'SEBRING 4C', 'COBALT', 'PT CRUISER 2.4L I4 S']
Trim (nominal, 135 distinct): ['Bas', 'LS', 'SE', 'SXT', 'LT', 'LX', 'Tou', 'EX', 'SEL', 'XLT']
SubModel (nominal, 864 distinct): ['4D SEDAN', '4D SEDAN LS', '4D SEDAN SE', '4D WAGON', 'MINIVAN 3.3L', '4D SUV 4.2L LS', '4D SEDAN LT', '4D SEDAN SXT FFV', '2D COUPE', '4D SEDAN LX']
Color (nominal, 17 distinct): ['SILVER', 'WHITE', 'BLUE', 'GREY', 'BLACK', 'RED', 'GOLD', 'GREEN', 'MAROON', 'BEIGE']
Transmission (nominal, 4 distinct): ['AUTO', 'MANUAL', 'Manual']
WheelTypeID (nominal, 5 distinct): ['1', '2', '3', '0']
WheelType (nominal, 4 distinct): ['Alloy', 'Covers', 'Special']
VehOdo (numeric, 39947 distinct): ['75009.0', '77995.0', '75371.0', '71225.0', '79015.0', '67464.0', '85884.0', '75786.0', '88958.0', '71823.0']
Nationality (nominal, 5 distinct): ['AMERICAN', 'OTHER ASIAN', 'TOP LINE ASIAN', 'OTHER']
Size (nominal, 13 distinct): ['MEDIUM', 'LARGE', 'MEDIUM SUV', 'COMPACT', 'VAN', 'LARGE TRUCK', 'SMALL SUV', 'SPECIALTY', 'CROSSOVER', 'LARGE SUV']
TopThreeAmericanName (nominal, 5 distinct): ['GM', 'CHRYSLER', 'FORD', 'OTHER']
MMRAcquisitionAuctionAveragePrice (numeric, 10343 distinct): ['0.0', '5480.0', '5569.0', '6311.0', '6858.0', '7644.0', '4573.0', '8196.0', '6892.0', '7048.0']
MMRAcquisitionAuctionCleanPrice (numeric, 11380 distinct): ['0.0', '6461.0', '6584.0', '7450.0', '8107.0', '1.0', '8892.0', '9044.0', '5967.0', '7614.0']
MMRAcquisitionRetailAveragePrice (numeric, 12726 distinct): ['0.0', '6418.0', '6515.0', '7316.0', '7907.0', '8756.0', '9352.0', '5439.0', '11114.0', '7943.0']
MMRAcquisitonRetailCleanPrice (numeric, 13457 distinct): ['0.0', '7478.0', '7611.0', '8546.0', '9256.0', '10103.0', '10268.0', '6944.0', '11562.0', '9613.0']
MMRCurrentAuctionAveragePrice (numeric, 10316 distinct): ['0.0', '5480.0', '5569.0', '6311.0', '8186.0', '7269.0', '6858.0', '7644.0', '8196.0', '8033.0']
MMRCurrentAuctionCleanPrice (numeric, 11266 distinct): ['0.0', '6461.0', '6584.0', '7450.0', '8107.0', '1.0', '8892.0', '7898.0', '9279.0', '9044.0']
MMRCurrentRetailAveragePrice (numeric, 12494 distinct): ['0.0', '6418.0', '6515.0', '7316.0', '7907.0', '11674.0', '8756.0', '9352.0', '10834.0', '5439.0']
MMRCurrentRetailCleanPrice (numeric, 13193 distinct): ['0.0', '7478.0', '7611.0', '8546.0', '9256.0', '10103.0', '10268.0', '12864.0', '11413.0', '12387.0']
PRIMEUNIT (nominal, 3 distinct): ['NO', 'YES']
AUCGUART (nominal, 3 distinct): ['GREEN', 'RED']
BYRNO (nominal, 74 distinct): ['99761', '18880', '835', '3453', '22916', '21053', '19619', '99750', '17675', '20928']
VNZIP1 (nominal, 153 distinct): ['32824', '27542', '75236', '74135', '80022', '85226', '85040', '29697', '95673', '28273']
VNST (nominal, 37 distinct): ['TX', 'FL', 'CA', 'NC', 'AZ', 'CO', 'SC', 'OK', 'GA', 'TN']
VehBCost (numeric, 2011 distinct): ['7500.0', '6500.0', '7200.0', '6000.0', '4200.0', '7000.0', '8000.0', '7800.0', '7100.0', '7400.0']
IsOnlineSale (nominal, 2 distinct): ['0', '1']
WarrantyCost (numeric, 281 distinct): ['920.0', '1974.0', '2152.0', '1389.0', '1215.0', '1155.0', '803.0', '728.0', '1503.0', '1086.0']
'''

CONTEXT = "Auto Dealership Kicked Cars Prediction"
TARGET = CuratedTarget(raw_name="IsBadBuy", new_name="Bad Buy", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'Bad', '1': 'Good'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="PurchDate", new_name="Purchase Date", feat_type=FeatureType.DATE)]