from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask
from tabular.preprocessing.textual import remove_commas

'''
Dataset Name: sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv
====
Examples: 37814
====
URL: https://www.kaggle.com/sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv
====
Description: 
Used Cars Dataset (CarDekho)
A dataset of used cars with all of their details and listing price.

About Dataset
This dataset contains information about ~38000 cars listed on Cardekho.

There are three CSV files in this dataset -

cars_overview.csv : Overview of the cars, contains basic info about the cars such as transmission type, location and the listing price.
car_details.csv : This file contains the almost all the cars in the overview file along with many other details, such as the features of the cars, the type of owner, etc.
car_details_merges.csv : This file is the merged version of the above two files, contains the basic as well as the detailed information of all the cars.
feature_dictionary.csv : Since the data is quite big, this file explains what information each column in the dataset has.

Points to note about the data:

The dataset contains columns which can have duplicate information since the data is scrapped using an API. It is advised to clean the data before using it.
There are multiple unique identifiers for each car, but using usedCarSkuId is recommended.
The price of the cars is under the column named price which has values such as ₹ 3.50 Lakh. We also have another column indicating the price in a continuous variable type called pu
Disclaimer
This data is meant for academic and research purposes and should not be used for commercial purposes.

====
Features:

position (int64, 20 distinct): ['19', '20', '10', '14', '18', '11', '12', '16', '8', '15']
loc (object, 511 distinct): ['Pune City', 'Gurgaon', 'Bangalore City', 'New Delhi G.P.O.', 'pune city', 'gurgaon', 'new delhi g.p.o.', 'bangalore city', 'Mahadevapura', 'Noida']
myear (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
bt (object, 11 distinct): ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Minivans', 'Luxury Vehicles', 'Pickup Trucks', 'Convertibles', 'Coupe', 'Wagon']
tt (object, 2 distinct): ['Manual', 'Automatic']
ft (object, 5 distinct): ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
km (object, 23863 distinct): ['70,000', '1,20,000', '80,000', '60,000', '90,000', '50,000', '40,000', '1,10,000', '1,00,000', '35,000']
ip (int64, 2 distinct): ['0', '1']
pi (object, 37134 distinct): ['https://images10.gaadi.com/usedcar_image/original/usedcar_0_734841676968137_1676968151.jpeg?imwidth=640', 'https://gcd73715.gaadi.com/user/server/php/files72/img2703774/usedcar_43_727151677065118_1677065130.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/spyne_VCC_car_replace_bg_53362fdf-2218-4dd6-8170-bed6c04d23b0_1693918_116_1676196764.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/zz1_crop_img_63f5b91f558ce_1677048095.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/spyne_VCC_car_replace_bg_842547b6-779e-4ed1-b95a-4f283baaf2f1_1693811_116_1676186532.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/spyne_VCC_car_replace_bg_b7b8d1d8-e111-407f-bbf5-04b0eb96ad55_1693730_116_1676183299.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/VCC_car_replace_bg_30892a27-d483-4571-957b-0f32af6726e6_1693219_94_1676101173.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/VCC_1693162_94_1676101837.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/VCC_1692794_94_1676026254.jpg?imwidth=640', 'https://images10.gaadi.com/usedcar_image/original/VCC_car_replace_bg_31e405f2-ffc5-4799-a02c-4cc35772031e_1692917_334_1676033126.jpg?imwidth=640']
images (object, 37135 distinct): ["[{'img': ''}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/usedcar_0_734841676968137_1676968151.jpeg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/VCC_car_replace_bg_31e405f2-ffc5-4799-a02c-4cc35772031e_1692917_334_1676033126.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/spyne_VCC_car_replace_bg_53362fdf-2218-4dd6-8170-bed6c04d23b0_1693918_116_1676196764.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/zz1_crop_img_63f5b91f558ce_1677048095.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/spyne_VCC_car_replace_bg_842547b6-779e-4ed1-b95a-4f283baaf2f1_1693811_116_1676186532.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/spyne_VCC_car_replace_bg_b7b8d1d8-e111-407f-bbf5-04b0eb96ad55_1693730_116_1676183299.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/VCC_car_replace_bg_30892a27-d483-4571-957b-0f32af6726e6_1693219_94_1676101173.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/VCC_1693162_94_1676101837.jpg?imwidth=640'}]", "[{'img': 'https://images10.gaadi.com/usedcar_image/original/VCC_1692794_94_1676026254.jpg?imwidth=640'}]"]
imgCount (int64, 54 distinct): ['15', '20', '21', '10', '11', '22', '1', '16', '12', '17']
threesixty (bool, 2 distinct): ['0', '1']
dvn (object, 4159 distinct): ['Maruti Swift VXI', 'Maruti Alto 800 LXI', 'Maruti Wagon R LXI CNG', 'Maruti Wagon R VXI BS IV', 'Maruti Swift VDI BSIV', 'Maruti Swift Dzire VDI', 'Honda City 1.5 S MT', 'Hyundai Grand i10 Sportz', 'Maruti Swift Dzire VXI', 'Hyundai i10 Magna']
oem (object, 46 distinct): ['Maruti', 'Hyundai', 'Honda', 'Mahindra', 'Tata', 'Toyota', 'Ford', 'Renault', 'Volkswagen', 'Skoda']
model (object, 382 distinct): ['Honda City', 'Hyundai i20', 'Maruti Swift', 'Maruti Wagon R', 'Maruti Swift Dzire', 'Hyundai i10', 'Hyundai Grand i10', 'Hyundai Creta', 'Hyundai Verna', 'Maruti Baleno']
modelId (int64, 682 distinct): ['627', '614', '586', '245', '262', '626', '255', '220', '992', '249']
vid (object, 4159 distinct): ['Maruti Swift VXI', 'Maruti Alto 800 LXI', 'Maruti Wagon R LXI CNG', 'Maruti Wagon R VXI BS IV', 'Maruti Swift VDI BSIV', 'Maruti Swift Dzire VDI', 'Honda City 1.5 S MT', 'Hyundai Grand i10 Sportz', 'Maruti Swift Dzire VXI', 'Hyundai i10 Magna']
centralVariantId (int64, 4585 distinct): ['4312', '1245', '4164', '7084', '4310', '1568', '3962', '1549', '3943', '1570']
variantName (object, 3488 distinct): ['VXI', 'LXI', 'Sportz', 'VDI', 'Magna', 'LXI CNG', 'VXI BS IV', 'VDI BSIV', 'Sportz 1.2', '1.5 S MT']
city_x (object, 617 distinct): ['New Delhi', 'Bangalore', 'Pune', 'Gurgaon', 'Mumbai', 'Hyderabad', 'Noida', 'Ahmedabad', 'Kolkata', 'Ghaziabad']
vlink (object, 37814 distinct): ['/buy-used-car-details/used-Maruti-Wagon-R-Lxi-Cng-cars-Lucknow_7111bf25-97af-47f9-867b-40879190d800.htm', '/used-car-details/used-Hyundai-I20-1.2-Spotz-cars-Bangalore_81b2e887-721f-4516-ae57-8e1d4bd1b735.htm', '/buy-used-car-details/used-Hyundai-Verna-1.6-Sx-Vtvt-cars-Bangalore_9760486f-1d97-49b1-896f-021457828c83.htm', '/buy-used-car-details/used-Hyundai-I10-Sportz-1.1l-cars-Bangalore_fe73c0a3-8f17-4877-9206-0796e618cef0.htm', '/used-car-details/used-Hyundai-Verna-1.6-Vtvt-At-Sx-cars-Bangalore_e4698829-42f5-41d8-b7cf-fceade705858.htm', '/used-car-details/used-Hyundai-Verna-1.6-Sx-Vtvt-(o)-cars-Bangalore_57d50729-c8a6-42f7-8c1a-5a41d0df2f07.htm', '/used-car-details/used-Hyundai-Verna-1.6-Ex-Vtvt-cars-Bangalore_4c55346a-2c62-459c-8bfb-98f1e4272383.htm', '/used-car-details/used-Hyundai-Grand-I10-Asta-cars-Bangalore_01dd086d-77ee-4dc7-95bc-e3d0ccf13df3.htm', '/used-car-details/used-Hyundai-I20-Asta-Option-1.2-cars-Bangalore_163d917b-8797-4764-819f-5cdd66ec971b.htm', '/used-car-details/used-Hyundai-I20-Sportz-Plus-Bsiv-cars-Bangalore_24eec1b8-85fd-4865-a71a-9d7b550542f2.htm']
price (object, 2374 distinct): ['₹ 3 Lakh', '₹ 3.50 Lakh', '₹ 4 Lakh', '₹ 5 Lakh', '₹ 2.50 Lakh', '₹ 4.50 Lakh', '₹ 6 Lakh', '₹ 2 Lakh', '₹ 5.50 Lakh', '₹ 6.50 Lakh']
pu (object, 6865 distinct): ['3,00,000', '3,50,000', '4,00,000', '5,00,000', '2,50,000', '4,50,000', '6,00,000', '2,00,000', '5,50,000', '6,50,000']
discountValue (int64, 30 distinct): ['0', '5000', '100000', '3000', '4000', '60000', '7000', '10000', '200000', '30000']
msp (int64, 3 distinct): ['0', '1093000', '959000']
priceSaving (object, 2 distinct): ['₹94K Regular Off', '₹60K Regular Off']
pageNo (int64, 162 distinct): ['2', '53', '54', '56', '55', '52', '57', '62', '61', '60']
utype (object, 2 distinct): ['Dealer', 'Individual']
views (int64, 2143 distinct): ['37', '47', '45', '39', '36', '35', '38', '22', '28', '25']
usedCarId (int64, 37814 distinct): ['3369178', '3368388', '3371014', '3370898', '3370861', '3370789', '3370592', '3370503', '3370492', '3370433']
usedCarSkuId (object, 37814 distinct): ['7111bf25-97af-47f9-867b-40879190d800', '81b2e887-721f-4516-ae57-8e1d4bd1b735', '9760486f-1d97-49b1-896f-021457828c83', 'fe73c0a3-8f17-4877-9206-0796e618cef0', 'e4698829-42f5-41d8-b7cf-fceade705858', '57d50729-c8a6-42f7-8c1a-5a41d0df2f07', '4c55346a-2c62-459c-8bfb-98f1e4272383', '01dd086d-77ee-4dc7-95bc-e3d0ccf13df3', '163d917b-8797-4764-819f-5cdd66ec971b', '24eec1b8-85fd-4865-a71a-9d7b550542f2']
ucid (int64, 37811 distinct): ['3590794', '3551965', '3583783', '3624006', '3623895', '3623771', '3623731', '3623653', '3623425', '3623335']
sid (object, 37811 distinct): ['170C604765B72FF5227D01C5F563DD91', '7B69CA452F143F1D77D71153ADF600A6', 'D18D7661F6467A006E48BB8BB1904151', '1079440731B5DE8B0F7492958FF21023', 'D41E1237CB02B0F21E48F4A5E12440BB', '055BFFFA64F995A6A34B86C7F8FB8D29', 'A016E19799188B82570BF31372295CE7', 'BE2C55134BFCA0623564FE3F96D40247', '08F3AD9A570B095087189D7179F2C8B6', '8A554B9B9E56DD829C67650567F676F7']
tmGaadiStore (bool, 2 distinct): ['0', '1']
emiwidget (object, 6206 distinct): ['{}', "{'title': 'EMI starts', 'cost': '12,411', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '14,893', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '16,134', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '13,652', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '17,375', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '11,170', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '18,616', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '19,857', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '9,929', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}"]
transmissionType (object, 2 distinct): ['Manual', 'Automatic']
dynx_itemid_x (int64, 37814 distinct): ['3369178', '3368388', '3371014', '3370898', '3370861', '3370789', '3370592', '3370503', '3370492', '3370433']
dynx_itemid2_x (int64, 4585 distinct): ['4312', '1245', '4164', '7084', '4310', '1568', '3962', '1549', '3943', '1570']
dynx_totalvalue_x (int64, 6865 distinct): ['300000', '350000', '400000', '500000', '250000', '450000', '600000', '200000', '550000', '650000']
leadForm (int64, 2 distinct): ['1', '0']
leadFormCta (object, 1 distinct): ['View Seller Details']
offers (object, 1 distinct): ['{}']
compare (bool, 1 distinct): ['1']
brandingIcon (object, 1 distinct): ['https://images10.gaadi.com/listing/icons/CertifiedV1.svg']
pageType (object, 2 distinct): ['cls', 'ucr']
carType (object, 3 distinct): ['partner', 'corporate', 'assured']
corporateId (int64, 6 distinct): ['7', '17', '13', '11', '16', '15']
top_features (object, 400 distinct): ["['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Centeral Locking', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Centeral Locking', 'Radio']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Brake Assist', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Brake Assist', 'Radio']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Power Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Centeral Locking', 'Radio']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Power Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Brake Assist', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Centeral Locking', 'Power Door Locks', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Power Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Centeral Locking', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Centeral Locking', 'Child Safety Locks']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Centeral Locking']"]
comfort_features (object, 2016 distinct): ["['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Vanity Mirror', 'Rear Seat Headrest', 'Cup Holders Front']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Air Quality Control', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Reading Lamp', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Height Adjustable Front Seat Belts', 'Cup Holders Front', 'Cup Holders Rear', 'Seat Lumbar Support', 'Multifunction Steering Wheel', 'Cruise Control', 'Rear ACVents']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Vanity Mirror', 'Rear Seat Headrest', 'Cup Holders Front', 'Seat Lumbar Support']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Air Quality Control', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Cup Holders Front', 'Cup Holders Rear', 'Multifunction Steering Wheel', 'Cruise Control', 'Rear ACVents']", "['Power Steering', 'Power Windows Front', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Rear Seat Headrest', 'Cup Holders Front']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Reading Lamp', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Cup Holders Front', 'Cup Holders Rear', 'Multifunction Steering Wheel']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Reading Lamp', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Height Adjustable Front Seat Belts', 'Cup Holders Front', 'Cup Holders Rear', 'Seat Lumbar Support', 'Multifunction Steering Wheel']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Vanity Mirror', 'Rear Seat Headrest', 'Cup Holders Front', 'Multifunction Steering Wheel']", "['Power Steering', 'Power Windows Front', 'Low Fuel Warning Light', 'Rear Seat Headrest', 'Cup Holders Front']", '[]']
interior_features (object, 527 distinct): ["['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Leather Steering Wheel', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Leather Steering Wheel', 'Glove Compartment', 'Digital Clock', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Leather Seats', 'Leather Steering Wheel', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Cigarette Lighter']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Height Adjustable Driver Seat', 'Dual Tone Dashboard']"]
exterior_features (object, 1893 distinct): ["['Adjustable Head Lights', 'Fog Lights Front', 'Power Adjustable Exterior Rear View Mirror', 'Electric Folding Rear View Mirror', 'Rear Window Wiper', 'Rear Window Washer', 'Rear Window Defogger', 'Alloy Wheels', 'Integrated Antenna', 'Tinted Glass', 'Rear Spoiler', 'Outside Rear View Mirror Turn Indicators', 'Chrome Grille', 'Chrome Garnish', 'Smoke Headlamps', 'Roof Rail']", "['Adjustable Head Lights', 'Fog Lights Front', 'Power Adjustable Exterior Rear View Mirror', 'Electric Folding Rear View Mirror', 'Rear Window Defogger', 'Alloy Wheels', 'Power Antenna', 'Outside Rear View Mirror Turn Indicators', 'Chrome Grille', 'Chrome Garnish']", '[]', "['Adjustable Head Lights', 'Fog Lights Front', 'Fog Lights Rear', 'Power Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Power Antenna', 'Tinted Glass', 'Outside Rear View Mirror Turn Indicators']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Power Antenna', 'Chrome Grille']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Halogen Headlamps']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Power Antenna', 'Chrome Grille', 'Chrome Garnish', 'Roof Rail']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Power Antenna', 'Chrome Grille', 'Halogen Headlamps']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Power Antenna']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Integrated Antenna', 'Tinted Glass']"]
safety_features (object, 2116 distinct): ["['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Crash Sensor', 'Ebd', 'Rear Camera', 'Anti Theft Device']", "['Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Anti Theft Device']", "['Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Anti Theft Device']", "['Anti Lock Braking System', 'Brake Assist', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Engine Immobilizer', 'Engine Check Warning']", "['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Crash Sensor', 'Ebd', 'Anti Theft Device']", '[]', "['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Crash Sensor', 'Ebd', 'Follow Me Home Headlamps', 'Rear Camera', 'Anti Theft Device', 'Impact Sensing Auto Door Lock']", "['Centeral Locking', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer']", "['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Engine Check Warning', 'Crash Sensor', 'Ebd', 'Anti Theft Device', 'Isofix Child Seat Mounts', 'Pretensioners And Force Limiter Seatbelts', 'No Of Airbags']", "['Centeral Locking', 'Child Safety Locks', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Engine Check Warning', 'Anti Theft Device']"]
Color (object, 798 distinct): ['White', 'Silver', 'Grey', 'Red', 'Blue', 'Black', 'Brown', 'Other', 'Golden', 'Gray']
Engine Type (object, 658 distinct): ['In-Line Engine', 'Kappa VTVT Petrol Engine', 'Petrol Engine', 'DDiS Diesel Engine', 'K Series Petrol Engine', 'Diesel Engine', 'i-VTEC Petrol Engine', 'mHawk Diesel Engine', 'VTVT Petrol Engine', 'i VTEC Engine']
Displacement (float64, 187 distinct): ['1197.0', '1248.0', '998.0', '1497.0', '1498.0', '1199.0', '1198.0', '2179.0', '999.0', '1086.0']
Max Power (object, 1018 distinct): ['81.80bhp@6000rpm', '88.5bhp@4000rpm', '74bhp@4000rpm', '81.86bhp@6000rpm', '98.6bhp@3600rpm', '140bhp@3750rpm', '78.9bhp@6000rpm', '117.3bhp@6600rpm', '82bhp@6000rpm', '121.3bhp@6400rpm']
Max Torque (object, 844 distinct): ['200Nm@1750rpm', '90Nm@3500rpm', '113Nm@4200rpm', '114Nm@4000rpm', '190Nm@2000rpm', '145Nm@4600rpm', '110Nm@4800rpm', '109Nm@4500rpm', '115Nm@4000rpm', '69Nm@3500rpm']
No of Cylinder (float64, 11 distinct): ['4.0', '3.0', '6.0', '5.0', '2.0', '7.0', '8.0', '1.0', '12.0', '10.0']
Values per Cylinder (float64, 7 distinct): ['4.0', '2.0', '3.0', '5.0', '1.0', '48.0', '8.0']
Value Configuration (object, 13 distinct): ['DOHC', 'SOHC', 'DOHC ', 'undefined', 'iDSI', 'DOHC with VIS', 'DOHC with VGT', '16 Modules 48 Cells', '16-valve DOHC layout', 'DOHC with TIS']
BoreX Stroke (object, 224 distinct): ['69.6 X 82 mm', '73.0 X 89.4 mm', '69 x 72 mm', '71 x 75.6 mm', '69.6 X 82', '73 X 82 mm', '73 X 71.5 mm', '73 x 72 mm', '77 X 85.8 mm', '73.5 X 88.3 mm']
Turbo Charger (object, 9 distinct): ['No', 'Yes', 'Twin', 'YES', 'NO', 'no', 'yes', 'twin', 'Turbo']
Super Charger (object, 5 distinct): ['No', 'Yes', 'NO', 'yes', 'no']
Length (object, 516 distinct): ['3995mm', '4440mm', '3765mm', '3985mm', '3585mm', '3840mm', '4585mm', '4270mm', '4315mm', '3990mm']
Width (object, 394 distinct): ['1695mm', '1735mm', '1680mm', '1660mm', '1595mm', '1495mm', '1734mm', '1745mm', '1475mm', '1699mm']
Height (object, 423 distinct): ['1505mm', '1530mm', '1475mm', '1520mm', '1495mm', '1700mm', '1550mm', '1640mm', '1510mm', '1785mm']
Wheel Base (object, 328 distinct): ['2450mm', '2600mm', '2425mm', '2380mm', '2570mm', '2360mm', '2400mm', '2520mm', '2700mm', '2430mm']
Front Tread (object, 219 distinct): ['1295mm', '1480mm', '1505mm', '1400mm', '1479mm', '1530mm', '1485mm', '1470mm', '1495mm', '1560mm']
Rear Tread (object, 238 distinct): ['1290mm', '1493mm', '1465mm', '1503mm', '1385mm', '1495mm', '1480mm', '1520mm', '1505mm', '1530mm']
Kerb Weight (object, 831 distinct): ['1066kg', '935kg', '885kg', '870kg', '860kg', '960kg', '1100kg', '1060kg', '855-885', '1025kg']
Gross Weight (object, 436 distinct): ['1350kg', '1340kg', '2510kg', '1680kg', '1185kg', '1415kg', '1505kg', '1315kg', '2450kg', '1490kg']
Gear Box (object, 132 distinct): ['5 Speed', '6 Speed', '5-Speed', '7 Speed', '5 Speed ', '6-Speed', '6 Speed ', '8 Speed', '4 Speed', '5']
Drive Type (object, 24 distinct): ['FWD', 'RWD', 'AWD', '2WD', '4WD', '2 WD', '4X2', 'FWD ', '4X4', 'Front Wheel Drive']
Seating Capacity (float64, 11 distinct): ['5.0', '7.0', '8.0', '4.0', '6.0', '9.0', '2.0', '10.0', '0.0', '13.0']
Steering Type (object, 11 distinct): ['Power', 'Electric', 'Manual', 'Electronic', 'Electrical', 'power', 'EPAS', 'Hydraulic', 'electric', 'MT']
Turning Radius (object, 239 distinct): ['5.3 metres', '5.2 metres', '4.8 metres', '4.7 metres', '4.6 metres', '5.4 metres', '5.6 metres', '4.9 meters', '4.9 metres', '4.5 metres']
Front Brake Type (object, 43 distinct): ['Disc', 'Ventilated Disc', 'Disc ', 'Solid Disc', 'Ventilated Discs', 'Disc & Caliper Type', 'Disk', 'Ventilated Disc ', 'Ventilated Disk', 'Ventilated discs']
Rear Brake Type (object, 48 distinct): ['Drum', 'Disc', 'Ventilated Disc', 'Solid Disc', 'Self-Adjusting Drum', 'Discs', 'Disc & Caliper Type', 'Leading-Trailing Drum', 'Ventilated Discs', 'Leading & Trailing Drum']
Top Speed (object, 372 distinct): ['165 Kmph', '170 Kmph', '180 Kmph', '160 Kmph', '195 Kmph', '190 Kmph', '172 kmph', '190 kmph', '160 kmph', '150 Kmph']
Acceleration (object, 414 distinct): ['10 Seconds', '15 Seconds', '14 Seconds', '19 Seconds', '12.9 Seconds', '18.6 Seconds', '13.3 Seconds', '12.36 seconds', '13.2 Seconds', '12.6 Seconds']
Tyre Type (object, 37 distinct): ['Tubeless,Radial', 'Tubeless', 'Tubeless, Radial', 'Tubeless Tyres', 'Radial', 'Tubeless,Radial ', 'Radial, Tubeless', 'Tubeless Radial Tyres', 'Radial, Tubless', 'Tubeless Tyres, Radial']
No Door Numbers (float64, 5 distinct): ['5.0', '4.0', '3.0', '2.0', '6.0']
Cargo Volumn (object, 398 distinct): ['510-litres', '400-litres', '339-litres', '256-liters', '350', '475-litres', '295-litres', '180-liters', '460-litres', '328-litres']
model_type_new (object, 1 distinct): ['used']
originalLocation (float64, 0 distinct): []
page_title (object, 37814 distinct): ['Used Maruti Wagon R LXI CNG Car in Lucknow, 2016 Model (Id- 7111bf25-97af-47f9-867b-40879190d800) - Find Best Deals! | CarDekho.com', 'Used Hyundai I20 1.2 Spotz Car in Bangalore, 2018 Model (Id- 81b2e887-721f-4516-ae57-8e1d4bd1b735) - Find Best Deals! | CarDekho.com', 'Used Hyundai Verna 1.6 SX VTVT Car in Bangalore, 2017 Model (Id- 9760486f-1d97-49b1-896f-021457828c83) - Find Best Deals! | CarDekho.com', 'Used Hyundai I10 Sportz 1.1L Car in Bangalore, 2015 Model (Id- fe73c0a3-8f17-4877-9206-0796e618cef0) - Find Best Deals! | CarDekho.com', 'Used Hyundai Verna 1.6 VTVT AT SX Car in Bangalore, 2017 Model (Id- e4698829-42f5-41d8-b7cf-fceade705858) - Find Best Deals! | CarDekho.com', 'Used Hyundai Verna 1.6 SX VTVT (O) Car in Bangalore, 2015 Model (Id- 57d50729-c8a6-42f7-8c1a-5a41d0df2f07) - Find Best Deals! | CarDekho.com', 'Used Hyundai Verna 1.6 EX VTVT Car in Bangalore, 2011 Model (Id- 4c55346a-2c62-459c-8bfb-98f1e4272383) - Find Best Deals! | CarDekho.com', 'Used Hyundai Grand I10 Asta Car in Bangalore, 2014 Model (Id- 01dd086d-77ee-4dc7-95bc-e3d0ccf13df3) - Find Best Deals! | CarDekho.com', 'Used Hyundai I20 Asta Option 1.2 Car in Bangalore, 2016 Model (Id- 163d917b-8797-4764-819f-5cdd66ec971b) - Find Best Deals! | CarDekho.com', 'Used Hyundai I20 Sportz Plus BSIV Car in Bangalore, 2019 Model (Id- 24eec1b8-85fd-4865-a71a-9d7b550542f2) - Find Best Deals! | CarDekho.com']
compare_car_details (float64, 0 distinct): []
seller_type_new (object, 2 distinct): ['dealer', 'individual']
seating_capacity_new (float64, 10 distinct): ['5.0', '7.0', '8.0', '4.0', '6.0', '9.0', '2.0', '10.0', '13.0', '14.0']
transmission_type (object, 2 distinct): ['manual', 'automatic']
model_year_new (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
car_type_new (object, 1 distinct): ['used']
model_name (object, 382 distinct): ['honda city', 'hyundai i20', 'maruti swift', 'maruti wagon r', 'maruti swift dzire', 'hyundai i10', 'hyundai grand i10', 'hyundai creta', 'hyundai verna', 'maruti baleno']
model_id_new (int64, 382 distinct): ['125', '148', '338', '344', '339', '146', '143', '138', '161', '322']
oem_name (object, 46 distinct): ['maruti', 'hyundai', 'honda', 'mahindra', 'tata', 'toyota', 'ford', 'renault', 'volkswagen', 'skoda']
price_range_segment (object, 5 distinct): ['2lakh-5lakh', '5lakh-8lakh', '10+lakh', '0lakh-2lakh', '8lakh-10lakh']
dealer_id_new (int64, 1 distinct): ['0']
state (object, 33 distinct): ['maharashtra', 'karnataka', 'delhi', 'uttar pradesh', 'haryana', 'gujarat', 'telangana', 'west bengal', 'tamil nadu', 'rajasthan']
city_id_new (int64, 617 distinct): ['49', '105', '205', '74', '201', '8', '348', '51', '338', '349']
fuel_type (object, 5 distinct): ['petrol', 'diesel', 'cng', 'lpg', 'electric']
max_engine_capacity_new (float64, 187 distinct): ['1197.0', '1248.0', '998.0', '1497.0', '1498.0', '1199.0', '1198.0', '2179.0', '999.0', '1086.0']
transmission_type_new (object, 2 distinct): ['manual', 'automatic']
km_driven (int64, 23862 distinct): ['70000', '120000', '80000', '60000', '90000', '50000', '40000', '110000', '100000', '35000']
model_new (object, 382 distinct): ['Honda City', 'Hyundai i20', 'Maruti Swift', 'Maruti Wagon R', 'Maruti Swift Dzire', 'Hyundai i10', 'Hyundai Grand i10', 'Hyundai Creta', 'Hyundai Verna', 'Maruti Baleno']
vehicle_type_new (object, 1 distinct): ['car']
brand_name (object, 46 distinct): ['maruti', 'hyundai', 'honda', 'mahindra', 'tata', 'toyota', 'ford', 'renault', 'volkswagen', 'skoda']
engine_cc (object, 6 distinct): ['1000cc-2000cc', '500cc-1000cc', '2000cc-3000cc', '3000cc-4000cc', '4000cc-5000cc', '5000cc Plus']
fuel_type_new (object, 5 distinct): ['petrol', 'diesel', 'cng', 'lpg', 'electric']
car_segment (object, 11 distinct): ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Minivans', 'Luxury Vehicles', 'Pickup Trucks', 'Convertibles', 'Coupe', 'Wagon']
used_carid (int64, 37814 distinct): ['3369178', '3368388', '3371014', '3370898', '3370861', '3370789', '3370592', '3370503', '3370492', '3370433']
city_name_new (object, 617 distinct): ['new delhi', 'bangalore', 'pune', 'gurgaon', 'mumbai', 'hyderabad', 'noida', 'ahmedabad', 'kolkata', 'ghaziabad']
page_type (object, 1 distinct): ['used-car.detail']
city_y (object, 617 distinct): ['new delhi', 'bangalore', 'pune', 'gurgaon', 'mumbai', 'hyderabad', 'noida', 'ahmedabad', 'kolkata', 'ghaziabad']
engine_capacity_new (object, 6 distinct): ['1000cc-2000cc', '500cc-1000cc', '2000cc-3000cc', '3000cc-4000cc', '4000cc-5000cc', '5000cc Plus']
body_type_new (object, 12 distinct): ['Hatchback cars', 'Sedan cars', 'SUV cars', 'MUV cars', 'Minivans cars', 'Luxury Vehicles cars', 'Pickup Trucks cars', 'Convertibles cars', 'Coupe cars', ' cars']
owner_type_new (object, 6 distinct): ['first', 'second', 'third', 'fourth', 'fifth', 'unregistered car']
mileage_new (object, 627 distinct): ['18.9 kmpl', '17 kmpl', '18.6 kmpl', '18 kmpl', '20.36 kmpl', '24.3 kmpl', '16.8 kmpl', '16 kmpl', '21.21 kmpl', '15.1 kmpl']
dealer_id (int64, 1 distinct): ['0']
model_year (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
variant_name (object, 4131 distinct): ['maruti swift vxi', 'maruti alto 800 lxi', 'maruti wagon r lxi cng', 'maruti wagon r vxi bs iv', 'maruti swift dzire vxi', 'maruti swift dzire vdi', 'maruti swift vdi bsiv', 'honda city 1.5 s mt', 'hyundai grand i10 sportz', 'hyundai i10 magna']
price_segment (object, 5 distinct): ['2lakh-5lakh', '5lakh-8lakh', '10+lakh', '0lakh-2lakh', '8lakh-10lakh']
dynx_event (object, 1 distinct): ['dynRmktParamsReady']
dynx_pagetype (object, 1 distinct): ['offerdetail']
dynx_itemid_y (int64, 37814 distinct): ['3369178', '3368388', '3371014', '3370898', '3370861', '3370789', '3370592', '3370503', '3370492', '3370433']
dynx_itemid2_y (int64, 4585 distinct): ['4312', '1245', '4164', '7084', '4310', '1568', '3962', '1549', '3943', '1570']
dynx_totalvalue_y (int64, 6867 distinct): ['300000', '350000', '400000', '500000', '250000', '450000', '600000', '200000', '550000', '650000']
brand_new (object, 46 distinct): ['maruti', 'hyundai', 'honda', 'mahindra', 'tata', 'toyota', 'ford', 'renault', 'volkswagen', 'skoda']
variant_new (object, 4131 distinct): ['maruti swift vxi', 'maruti alto 800 lxi', 'maruti wagon r lxi cng', 'maruti wagon r vxi bs iv', 'maruti swift dzire vxi', 'maruti swift dzire vdi', 'maruti swift vdi bsiv', 'honda city 1.5 s mt', 'hyundai grand i10 sportz', 'hyundai i10 magna']
exterior_color (object, 798 distinct): ['White', 'Silver', 'Grey', 'Red', 'Blue', 'Black', 'Brown', 'Other', 'Golden', 'Gray']
min_engine_capacity_new (float64, 187 distinct): ['1197.0', '1248.0', '998.0', '1497.0', '1498.0', '1199.0', '1198.0', '2179.0', '999.0', '1086.0']
owner_type (object, 6 distinct): ['first', 'second', 'third', 'fourth', 'fifth', 'unregistered car']
price_segment_new (object, 5 distinct): ['2lakh-5lakh', '5lakh-8lakh', '10+lakh', '0lakh-2lakh', '8lakh-10lakh']
template_name_new (object, 4 distinct): ['used cardetail v2', 'used cardetail v2/corporate/13', 'used cardetail v2/corporate/16', 'used cardetail v2/ucr']
page_template (object, 1 distinct): ['Used Car > Detail Page']
template_Type_new (object, 1 distinct): ['used']
experiment (object, 1 distinct): ['control']
Fuel Suppy System (object, 99 distinct): ['MPFI', 'CRDi', 'CRDI', 'MPFi', 'Direct Injection', 'PGM-Fi', 'PGM - Fi', 'Common Rail', 'GDi', 'EFI(Electronic Fuel Injection)']
Compression Ratio (object, 100 distinct): ['17.6:1', '10.5:1', '10.3:1', '11.0:1', '16.0:1', '10.1:1', '16.5:1', '10.0:1', '9.0:1', '17.3:1']
Alloy Wheel Size (object, 18 distinct): ['16', '15', '14', '17', '13', '18', 'R16', '12', 'R17', '19']
Ground Clearance Unladen (object, 31 distinct): ['190mm', '209 mm', '178mm', '185mm', '192mm', '180mm', '120mm', '155mm', '160', '116mm']
'''

CONTEXT = "User cars and listing price in the website Cardekho"
TARGET = CuratedTarget(raw_name="pu", task_type=SupervisedTask.REGRESSION, processing_func=remove_commas)
COLS_TO_DROP = ['usedCarSkuId', 'price', 'price_segment_new', 'price_segment', 'ucid', 'sid', 'vlink',
                "price_range_segment", "used_carid", "pi"]
FEATURES = []

DESCRIPTION = '''
Used Cars Dataset (CarDekho)
A dataset of used cars with all of their details and listing price.

About Dataset
This dataset contains information about ~38000 cars listed on Cardekho.

There are three CSV files in this dataset -

cars_overview.csv : Overview of the cars, contains basic info about the cars such as transmission type, location and the listing price.
car_details.csv : This file contains the almost all the cars in the overview file along with many other details, such as the features of the cars, the type of owner, etc.
car_details_merges.csv : This file is the merged version of the above two files, contains the basic as well as the detailed information of all the cars.
feature_dictionary.csv : Since the data is quite big, this file explains what information each column in the dataset has.

Points to note about the data:

The dataset contains columns which can have duplicate information since the data is scrapped using an API. It is advised to clean the data before using it.
There are multiple unique identifiers for each car, but using usedCarSkuId is recommended.
The price of the cars is under the column named price which has values such as ₹ 3.50 Lakh. We also have another column indicating the price in a continuous variable type called pu
Disclaimer
This data is meant for academic and research purposes and should not be used for commercial purposes.
'''