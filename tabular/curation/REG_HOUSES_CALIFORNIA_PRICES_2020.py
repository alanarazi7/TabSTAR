from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: california_house_price
====
Examples: 37951
====
URL: https://www.openml.org/search?type=data&id=46669
====
Description: Predict sale prices of California homes sold in 2020 based on a text summary
    written by the seller and various tabular features (e.g. bedroom number, home type, location, year built, parking). Representing a regression task with many features that are
    text and numeric, this dataset originally stems from a 2021 Kaggle prediction competition:
    https://www.kaggle.com/c/california-house-prices
  
 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: Sold_Price (numeric, 4990 distinct): ['2.6327', '2.6389', '2.6446', '2.6418', '2.6293', '2.614', '2.6473', '2.6359', '2.6547', '2.6499']
====
Features:

Address (string, 37870 distinct): ['400 Mariners Island Blvd', '1 Appian Way', '10592 Boulders Rd', '12941 Riding Trail Dr', '424 Santa Monica Dr', '555 Fulton St STE 206', '2309 S West View St', '3309 Aria Ln', '270 Ridgeview Ct', '12422 Villa Ct #2']
Summary (string, 37454 distinct): ['Neighboring The Ace Hotel, 939 S. Broadway is the epitome of effort lessliving in Downtown Los Angeles. While the former Western Costume Building maintains its iconic exterior architecture, each residence has been renovated to feature the finest modern details. Stainless steel appliances and Quartz countertops add contemporary sophistication to the loft-style, industrial chic interiors. The original, towering windows, capture abundant natural light and reveal dynamic views of Downtown Los Angeles. One- and two-bedrooms ranging from 505 to 1,227 square feet, residences feature open-concept living and boast high ceilings. Providing generous gathering spaces, the interiors showcase the perfect balance of form and function. Amenities include rooftop pool/spa, cabanas, gym, in-unit stacked laundry, and high ceilings.', '-', '.', "BRAND NEW PROJECT! This is THE best adaptive re-use project in DTLA to date. 151 Live/Work units in the first historic condo building in South Park. Great variety of very attractive floor plans, with 11-14 ft ceilings, all with spectacular city views. LOWEST HOA'S IN DTLA. 1 COVERED PARKING SPACE INCLUDED in building across the street. Amazing amenities incl. rooftop sundeck w/ pool, spa, lounge chairs and cabanas, gym, and theater and media room on the basement floor, and beautiful historic lobby w/ security guard. Former home of the Western Costume Co est. 1924. Adjacent to the United Artists Theater (and now the Ace Hotel) 939 South Broadway was the largest costume rental house in the world, responsible for 99% of the costumes used in all Western movies in the early days of Hollywood. Bldg was the vision of big screen stars Mary Pickford, Charlie Chaplin and Douglas Fairbanks of the United Artists studio. AGENTS SEE PRIVATE AND SHOWING REMARKS!", '40 units remaining! Many floor plans available. BRAND NEW CONSTRUCTION! LARGE living rooms and LARGE bedrooms with great views of DTLA, Koreatown & Hollywood. Quartz Countertops, Thor Appliances, Hardwood Flooring & very spacious laundry rooms. Two assigned SIDE BY SIDE parking spaces in secured garage, bicycle storage & Electric vehicle charging stations. On-site fitness room, Pool, Lounge/Club Room, 2 courtyards and  20 guest parking. Conveniently located to Metro Station on Vermont and Wilshire. Plenty of restaurants and entertainment within walking distance. Minutes from DTLA. Pictures may be photos of a different unit. 3D  https://my.matterport.com/show/?m=T8XeseJaweX/  Sales Office OPEN DAILY from 11-5pm', 'Murieta Gardens is an all electric community offering 4 single-story new home designs in Rancho Murieta, CA. Live a relaxed lifestyle in a great school district with easy commute routes to Elk Grove, Downtown Sacramento and Folsom. Offered By: K. Hovnanian Homes Northern California, Inc.', '***STAY TUNED FOR MEGA OPEN HOUSE INFO, PICS, VIDEO TOUR AND DETAILS.***', 'This two-bedroom home features everything you need on one level. A den offers versatility and the stylish kitchen with center island enjoys an open layout that effortlessly flows into the dining area and family room. Both bedrooms flank the living area for added privacy and each one enjoys a walk-in closet, while the master features a luxurious bathroom and upper deck access.Virtual Tour', 'Brand new townhouses in a gated community built in 2019. Contemporary Mediterranean design with many goodies; including granite and quartz counter-tops, glass tile back-splash, fireplace, skylight, central HVAC, tank-less water heater, private indoor laundry, walk-in closets, master suite, high ceilings, private patio, and much more. All units come equipped with stainless steel kitchen appliances including dishwasher, stove, refrigerator, range, and microwave. Community amenities include BBQ area, playground, spacious community building, remote entry, community garden, and more. This new development offers eligible buyers down payment assistance. All buyers but complete a new homebuyer education course. Please call for additional details.', 'For Comp Purposes Only...']
Type (string, 134 distinct): ['SingleFamily', 'Condo', 'Townhouse', 'Unknown', 'MultiFamily', 'MobileManufactured', 'VacantLand', 'Single Family', 'Apartment', 'Residential Lot']
Year_built (numeric, 161 distinct): ['1973.0', '2020.0', '1950.0', '2006.0', '1972.0', '1955.0', '1924.0', '1948.0', '1963.0', '1979.0']
Heating (string, 1593 distinct): ['Central', 'Central Forced Air', 'Central Forced Air - Gas', 'Forced Air', 'Wall Furnace', 'Other', 'Forced air, Gas', 'Forced air', 'Central, Gas', 'Gas']
Cooling (string, 472 distinct): ['Central Air', 'Central AC', 'Central', 'Ceiling Fan', 'Ceiling Fan, Central AC', 'Other', 'Wall/Window Unit(s)', 'Multi-Zone, Central AC', 'Window / Wall Unit', 'See Remarks']
Parking (string, 4341 distinct): ['Garage, Garage - Attached, Covered', '0 spaces', 'Garage, Garage - Attached', 'Garage', 'Garage - Attached', 'Garage - Attached, Covered', 'Carport', 'Covered', 'Driveway', 'Garage, Garage - Detached, Covered']
Lot (numeric, 7318 distinct): ['2.2531', '2.2738', '2.2808', '2.3355', '2.1775', '2.2935', '2.2721', '2.458', '2.1978', '2.1773']
Bedrooms (string, 249 distinct): ['3', '2', '4', '1', '5', 'Walk-in Closet', 'Master Suite / Retreat, Walk-in Closet', 'Master Suite / Retreat', 'Ground Floor Bedroom', '6']
Bathrooms (numeric, 24 distinct): ['2.0', '3.0', '1.0', '4.0', '5.0', '0.0', '6.0', '7.0', '8.0', '9.0']
Full_bathrooms (numeric, 17 distinct): ['2.0', '1.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
Total_interior_livable_area (numeric, 4340 distinct): ['1200.0', '1440.0', '1000.0', '1300.0', '1400.0', '1800.0', '1500.0', '1100.0', '1040.0', '2000.0']
Total_spaces (numeric, 48 distinct): ['2.0', '0.0', '1.0', '3.0', '4.0', '6.0', '5.0', '8.0', '10.0', '7.0']
Garage_spaces (numeric, 44 distinct): ['2.0', '0.0', '1.0', '3.0', '4.0', '6.0', '5.0', '8.0', '10.0', '7.0']
Region (string, 914 distinct): ['Los Angeles', 'San Jose', 'San Francisco', 'San Mateo', 'Sunnyvale', 'Santa Cruz', 'Santa Clara', 'Los Gatos', 'Morgan Hill', 'Mountain View']
Elementary_School (string, 1656 distinct): ['Laurel', 'Daniel Webster', 'Sherman', 'Kenter Canyon', 'Cabrillo', 'Gardner Street', 'Warner Avenue', 'Rooftop', 'McKinley', 'Westwood Charter']
Elementary_School_Score (numeric, 11 distinct): ['5.0', '6.0', '4.0', '7.0', '8.0', '9.0', '3.0', '2.0', '1.0', '10.0']
Elementary_School_Distance (numeric, 234 distinct): ['0.3', '0.4', '0.2', '0.5', '0.6', '0.7', '0.1', '0.8', '0.9', '1.0']
Middle_School (string, 482 distinct): ['Hubert Howe Bancroft Middle School', 'Mission Hill Middle School', 'Lick (James) Middle School', 'John Muir Middle School', 'Thomas Starr King Middle School', 'Willow Glen Middle School', 'Audubon Middle School', 'John Burroughs Middle School', 'C. T. English Middle School', 'Martin Murphy Middle School']
Middle_School_Score (numeric, 10 distinct): ['6.0', '3.0', '5.0', '7.0', '4.0', '8.0', '9.0', '2.0', '1.0']
Middle_School_Distance (numeric, 213 distinct): ['0.6', '0.8', '0.7', '0.5', '0.9', '0.4', '1.0', '0.3', '1.1', '1.2']
High_School (string, 619 distinct): ['Fairfax Senior High School', 'University Senior High School Charter', 'Harbor High School', 'Independence High School', 'John Marshall Senior High School', 'Willow Glen High School', 'Santa Clara High School', 'Belmont Senior High School', 'Aptos High School', 'Menlo-Atherton High School']
High_School_Score (numeric, 11 distinct): ['7.0', '6.0', '8.0', '5.0', '4.0', '3.0', '9.0', '2.0', '10.0', '1.0']
High_School_Distance (numeric, 333 distinct): ['0.7', '0.8', '0.9', '0.6', '1.0', '0.5', '1.1', '1.2', '1.3', '0.4']
Flooring (string, 1213 distinct): ['Wood', 'Hardwood', 'Laminate', 'Tile, Hardwood', 'Tile, Wood', 'Tile, Hardwood, Carpet', 'Wood, Tile', 'Carpet, Wood', 'Tile, Laminate', 'Tile, Carpet']
Heating_features (string, 964 distinct): ['Forced air, Gas', 'Central', 'Forced air', 'Other', 'Wall Furnace', 'Forced Air', 'Forced air, Electric', 'Central, Gas', 'Radiant', 'Wall']
Cooling_features (string, 276 distinct): ['Central', 'Central Air', 'Other', 'Wall', 'Wall/Window Unit(s)', 'Central, Solar', 'Evaporative', 'Central, Other', 'See Remarks', 'Central Air, Electric']
Appliances_included (string, 3939 distinct): ['Dishwasher, Dryer, Garbage disposal, Microwave, Range / Oven, Refrigerator, Washer', 'Dishwasher, Dryer, Freezer, Garbage disposal, Microwave, Range / Oven, Refrigerator, Washer', 'Dryer, Washer', 'Dishwasher', 'Dishwasher, Garbage disposal, Microwave, Range / Oven, Refrigerator', 'Dishwasher, Dryer, Garbage disposal, Range / Oven, Refrigerator, Washer', 'Range / Oven', 'Dishwasher, Dryer, Microwave, Range / Oven, Refrigerator, Washer', 'Dishwasher, Garbage disposal, Range / Oven, Refrigerator', 'Dishwasher, Garbage disposal, Microwave, Range / Oven']
Laundry_features (string, 1748 distinct): ['Inside', 'In Garage', 'Laundry Closet', 'Laundry Room', 'Washer / Dryer, Inside', 'Washer / Dryer', 'Inside Room', 'In Unit', 'In Garage, Washer / Dryer', 'Community']
Parking_features (string, 4212 distinct): ['Garage, Garage - Attached, Covered', 'Garage, Garage - Attached', 'Garage', 'Garage - Attached', 'Garage - Attached, Covered', 'Carport', 'Covered', 'Garage, Garage - Detached, Covered', 'Driveway', 'Garage - Two Door']
Tax_assessed_value (numeric, 28581 distinct): ['2.6899', '2.7092', '2.686', '2.6675', '2.724', '2.7003', '2.7021', '2.7077', '2.6492', '2.6774']
Annual_tax_amount (numeric, 17118 distinct): ['2.1778', '2.1092', '2.1692', '2.1505', '2.177', '2.2442', '2.1684', '2.1664', '2.2541', '2.0912']
Listed_On (string, 2248 distinct): ['2020-10-16', '2020-09-18', '2020-10-02', '2020-10-09', '2020-10-23', '2020-10-15', '2020-09-11', '2020-08-14', '2020-09-25', '2020-09-04']
Listed_Price (numeric, 4212 distinct): ['2.6181', '2.6094', '2.5995', '2.588', '2.6545', '2.6445', '2.6254', '2.6257', '2.6327', '2.6388']
Last_Sold_On (string, 5765 distinct): ['2015-02-27', '2016-09-30', '2017-06-30', '2018-05-25', '2016-07-29', '2019-08-30', '2013-08-30', '2019-04-30', '2017-12-29', '2019-05-31']
Last_Sold_Price (numeric, 3570 distinct): ['650000.0', '850000.0', '500000.0', '600000.0', '550000.0', '450000.0', '350000.0', '300000.0', '1100000.0', '750000.0']
City (string, 902 distinct): ['Los Angeles', 'San Jose', 'San Francisco', 'San Mateo', 'Santa Clara', 'Sunnyvale', 'Santa Cruz', 'Los Gatos', 'Morgan Hill', 'Redwood City']
Zip (numeric, 1206 distinct): ['95037', '95123', '95125', '95124', '95020', '90046', '95035', '94110', '95003', '90042']
State (string, 2 distinct): ['CA', 'AZ']
'''

CONTEXT = "California Houses 2020 Prices"
TARGET = CuratedTarget(raw_name="Sold_Price", new_name="Sold Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
