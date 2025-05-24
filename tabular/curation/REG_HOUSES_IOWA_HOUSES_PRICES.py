from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: house_prices
====
URL: https://www.openml.org/search?type=data&id=42165
====
Description: Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
ExterCond: Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)
====
Target Variable: SalePrice (numeric, 663 distinct): ['140000', '135000', '155000', '145000', '190000', '110000', '115000', '160000', '130000', '139000']
====
Features:

Id (numeric, 1460 distinct): ['1', '982', '980', '979', '978', '977', '976', '975', '974', '973']
MSSubClass (numeric, 15 distinct): ['20', '60', '50', '120', '30', '160', '70', '80', '90', '190']
MSZoning (string, 5 distinct): ['RL', 'RM', 'FV', 'RH', 'C (all)']
LotFrontage (numeric, 369 distinct): ['60.0', '70.0', '80.0', '50.0', '75.0', '65.0', '85.0', '78.0', '21.0', '90.0']
LotArea (numeric, 1073 distinct): ['7200', '9600', '6000', '9000', '8400', '10800', '1680', '7500', '9100', '8125']
Street (string, 2 distinct): ['Pave', 'Grvl']
Alley (string, 3 distinct): ['Grvl', 'Pave']
LotShape (string, 4 distinct): ['Reg', 'IR1', 'IR2', 'IR3']
LandContour (string, 4 distinct): ['Lvl', 'Bnk', 'HLS', 'Low']
Utilities (string, 2 distinct): ['AllPub', 'NoSeWa']
LotConfig (string, 5 distinct): ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
LandSlope (string, 3 distinct): ['Gtl', 'Mod', 'Sev']
Neighborhood (string, 25 distinct): ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW']
Condition1 (string, 9 distinct): ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe']
Condition2 (string, 8 distinct): ['Norm', 'Feedr', 'Artery', 'RRNn', 'PosN', 'PosA', 'RRAn', 'RRAe']
BldgType (string, 5 distinct): ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2fmCon']
HouseStyle (string, 8 distinct): ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin']
OverallQual (numeric, 10 distinct): ['5', '6', '7', '8', '4', '9', '3', '10', '2', '1']
OverallCond (numeric, 9 distinct): ['5', '6', '7', '8', '4', '3', '9', '2', '1']
YearBuilt (numeric, 112 distinct): ['2006', '2005', '2004', '2007', '2003', '1976', '1977', '1920', '1959', '1998']
YearRemodAdd (numeric, 61 distinct): ['1950', '2006', '2007', '2005', '2004', '2000', '2003', '2002', '2008', '1996']
RoofStyle (string, 6 distinct): ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed']
RoofMatl (string, 8 distinct): ['CompShg', 'Tar&Grv', 'WdShngl', 'WdShake', 'Metal', 'Membran', 'Roll', 'ClyTile']
Exterior1st (string, 15 distinct): ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood', 'CemntBd', 'BrkFace', 'WdShing', 'Stucco', 'AsbShng']
Exterior2nd (string, 16 distinct): ['VinylSd', 'MetalSd', 'HdBoard', 'Wd Sdng', 'Plywood', 'CmentBd', 'Wd Shng', 'Stucco', 'BrkFace', 'AsbShng']
MasVnrType (string, 5 distinct): ['None', 'BrkFace', 'Stone', 'BrkCmn']
MasVnrArea (numeric, 335 distinct): ['0.0', '180.0', '72.0', '108.0', '120.0', '16.0', '200.0', '340.0', '106.0', '80.0']
ExterQual (string, 4 distinct): ['TA', 'Gd', 'Ex', 'Fa']
ExterCond (string, 5 distinct): ['TA', 'Gd', 'Fa', 'Ex', 'Po']
Foundation (string, 6 distinct): ['PConc', 'CBlock', 'BrkTil', 'Slab', 'Stone', 'Wood']
BsmtQual (string, 5 distinct): ['TA', 'Gd', 'Ex', 'Fa']
BsmtCond (string, 5 distinct): ['TA', 'Gd', 'Fa', 'Po']
BsmtExposure (string, 5 distinct): ['No', 'Av', 'Gd', 'Mn']
BsmtFinType1 (string, 7 distinct): ['Unf', 'GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ']
BsmtFinSF1 (numeric, 637 distinct): ['0', '24', '16', '686', '662', '20', '936', '616', '560', '553']
BsmtFinType2 (string, 7 distinct): ['Unf', 'Rec', 'LwQ', 'BLQ', 'ALQ', 'GLQ']
BsmtFinSF2 (numeric, 144 distinct): ['0', '180', '374', '551', '147', '294', '391', '539', '96', '480']
BsmtUnfSF (numeric, 780 distinct): ['0', '728', '384', '600', '300', '572', '270', '625', '672', '440']
TotalBsmtSF (numeric, 721 distinct): ['0', '864', '672', '912', '1040', '816', '768', '728', '894', '780']
Heating (string, 6 distinct): ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']
HeatingQC (string, 5 distinct): ['Ex', 'TA', 'Gd', 'Fa', 'Po']
CentralAir (string, 2 distinct): ['Y', 'N']
Electrical (string, 6 distinct): ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
1stFlrSF (numeric, 753 distinct): ['864', '1040', '912', '894', '848', '672', '630', '816', '483', '960']
2ndFlrSF (numeric, 417 distinct): ['0', '728', '504', '546', '672', '600', '720', '896', '862', '780']
LowQualFinSF (numeric, 24 distinct): ['0', '80', '360', '205', '479', '397', '514', '120', '481', '232']
GrLivArea (numeric, 861 distinct): ['864', '1040', '894', '1456', '848', '1200', '912', '816', '1092', '1728']
BsmtFullBath (numeric, 4 distinct): ['0', '1', '2', '3']
BsmtHalfBath (numeric, 3 distinct): ['0', '1', '2']
FullBath (numeric, 4 distinct): ['2', '1', '3', '0']
HalfBath (numeric, 3 distinct): ['0', '1', '2']
BedroomAbvGr (numeric, 8 distinct): ['3', '2', '4', '1', '5', '6', '0', '8']
KitchenAbvGr (numeric, 4 distinct): ['1', '2', '3', '0']
KitchenQual (string, 4 distinct): ['TA', 'Gd', 'Ex', 'Fa']
TotRmsAbvGrd (numeric, 12 distinct): ['6', '7', '5', '8', '4', '9', '10', '11', '3', '12']
Functional (string, 7 distinct): ['Typ', 'Min2', 'Min1', 'Mod', 'Maj1', 'Maj2', 'Sev']
Fireplaces (numeric, 4 distinct): ['0', '1', '2', '3']
FireplaceQu (string, 6 distinct): ['Gd', 'TA', 'Fa', 'Ex', 'Po']
GarageType (string, 7 distinct): ['Attchd', 'Detchd', 'BuiltIn', 'Basment', 'CarPort', '2Types']
GarageYrBlt (numeric, 178 distinct): ['2005.0', '2006.0', '2004.0', '2003.0', '2007.0', '1977.0', '1998.0', '1999.0', '1976.0', '2008.0']
GarageFinish (string, 4 distinct): ['Unf', 'RFn', 'Fin']
GarageCars (numeric, 5 distinct): ['2', '1', '3', '0', '4']
GarageArea (numeric, 441 distinct): ['0', '440', '576', '240', '484', '528', '288', '400', '264', '480']
GarageQual (string, 6 distinct): ['TA', 'Fa', 'Gd', 'Ex', 'Po']
GarageCond (string, 6 distinct): ['TA', 'Fa', 'Gd', 'Po', 'Ex']
PavedDrive (string, 3 distinct): ['Y', 'N', 'P']
WoodDeckSF (numeric, 274 distinct): ['0', '192', '100', '144', '120', '168', '140', '224', '208', '240']
OpenPorchSF (numeric, 202 distinct): ['0', '36', '48', '20', '40', '45', '24', '30', '60', '39']
EnclosedPorch (numeric, 120 distinct): ['0', '112', '96', '192', '144', '120', '216', '156', '116', '252']
3SsnPorch (numeric, 20 distinct): ['0', '168', '144', '180', '216', '290', '153', '96', '23', '162']
ScreenPorch (numeric, 76 distinct): ['0', '192', '120', '224', '189', '180', '147', '90', '160', '144']
PoolArea (numeric, 8 distinct): ['0', '512', '648', '576', '555', '480', '519', '738']
PoolQC (string, 4 distinct): ['Gd', 'Ex', 'Fa']
Fence (string, 5 distinct): ['MnPrv', 'GdPrv', 'GdWo', 'MnWw']
MiscFeature (string, 5 distinct): ['Shed', 'Gar2', 'Othr', 'TenC']
MiscVal (numeric, 21 distinct): ['0', '400', '500', '700', '450', '600', '2000', '1200', '480', '15500']
MoSold (numeric, 12 distinct): ['6', '7', '5', '4', '8', '3', '10', '11', '9', '12']
YrSold (numeric, 5 distinct): ['2009', '2007', '2006', '2008', '2010']
SaleType (string, 9 distinct): ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'ConLw', 'CWD', 'Oth', 'Con']
SaleCondition (string, 6 distinct): ['Normal', 'Partial', 'Abnorml', 'Family', 'Alloca', 'AdjLand']
'''

CONTEXT = "Prices of Houses in Ames, Iowa"
TARGET = CuratedTarget(raw_name='SalePrice', new_name='Sale Price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []
