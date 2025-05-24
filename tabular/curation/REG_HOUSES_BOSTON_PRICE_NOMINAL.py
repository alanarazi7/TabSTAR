from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: house_prices_nominal
====
Examples: 1460
====
URL: https://www.openml.org/search?type=data&id=42563
====
Description: **Author**: Kaggle  
**Source**: [original](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) - 2011  
**Please cite**: Dean De Cock (2011) Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project, Journal of Statistics Education, 19:3, DOI: 10.1080/10691898.2011.11889627  

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. 
    
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.

MSSubClass: The building class

MSZoning: The general zoning classification

LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access

Alley: Type of alley access

LotShape: General shape of property

LandContour: Flatness of the property

Utilities: Type of utilities available

LotConfig: Lot configuration

LandSlope: Slope of property

Neighborhood: Physical locations within Ames city limits

Condition1: Proximity to main road or railroad

Condition2: Proximity to main road or railroad (if a second is present)

BldgType: Type of dwelling

HouseStyle: Style of dwelling

OverallQual: Overall material and finish quality

OverallCond: Overall condition rating

YearBuilt: Original construction date

YearRemodAdd: Remodel date

RoofStyle: Type of roof

RoofMatl: Roof material

Exterior1st: Exterior covering on house

Exterior2nd: Exterior covering on house (if more than one material)

MasVnrType: Masonry veneer type

MasVnrArea: Masonry veneer area in square feet

ExterQual: Exterior material quality

ExterCond: Present condition of the material on the exterior

Foundation: Type of foundation

BsmtQual: Height of the basement

BsmtCond: General condition of the basement

BsmtExposure: Walkout or garden level basement walls

BsmtFinType1: Quality of basement finished area

BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Quality of second finished area (if present)

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating

HeatingQC: Heating quality and condition

CentralAir: Central air conditioning

Electrical: Electrical system

1stFlrSF: First Floor square feet

2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Number of bedrooms above basement level

Kitchen: Number of kitchens

KitchenQual: Kitchen quality

TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality rating

Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

GarageType: Garage location

GarageYrBlt: Year garage was built

GarageFinish: Interior finish of the garage

GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

GarageCond: Garage condition

PavedDrive: Paved driveway

WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality

Fence: Fence quality

MiscFeature: Miscellaneous feature not covered in other categories

MiscVal: $Value of miscellaneous feature

MoSold: Month Sold

YrSold: Year Sold

SaleType: Type of sale

SaleCondition: Condition of sale
====
Target Variable: SalePrice (numeric, 663 distinct): ['140000', '135000', '155000', '145000', '190000', '110000', '115000', '160000', '130000', '139000']
====
Features:

MSSubClass (numeric, 15 distinct): ['20', '60', '50', '120', '30', '160', '70', '80', '90', '190']
MSZoning (nominal, 5 distinct): ['RL', 'RM', 'FV', 'RH', 'C (all)']
LotFrontage (numeric, 111 distinct): ['60.0', '70.0', '80.0', '50.0', '75.0', '65.0', '85.0', '78.0', '21.0', '90.0']
LotArea (numeric, 1073 distinct): ['7200', '9600', '6000', '9000', '8400', '10800', '1680', '7500', '9100', '8125']
Street (nominal, 2 distinct): ['Pave', 'Grvl']
Alley (nominal, 3 distinct): ['Grvl', 'Pave']
LotShape (nominal, 4 distinct): ['Reg', 'IR1', 'IR2', 'IR3']
LandContour (nominal, 4 distinct): ['Lvl', 'Bnk', 'HLS', 'Low']
Utilities (nominal, 2 distinct): ['AllPub', 'NoSeWa']
LotConfig (nominal, 5 distinct): ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
LandSlope (nominal, 3 distinct): ['Gtl', 'Mod', 'Sev']
Neighborhood (nominal, 25 distinct): ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW']
Condition1 (nominal, 9 distinct): ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe']
Condition2 (nominal, 8 distinct): ['Norm', 'Feedr', 'Artery', 'PosN', 'RRNn', 'PosA', 'RRAe', 'RRAn']
BldgType (nominal, 5 distinct): ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2fmCon']
HouseStyle (nominal, 8 distinct): ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin']
OverallQual (numeric, 10 distinct): ['5', '6', '7', '8', '4', '9', '3', '10', '2', '1']
OverallCond (numeric, 9 distinct): ['5', '6', '7', '8', '4', '3', '9', '2', '1']
YearBuilt (numeric, 112 distinct): ['2006', '2005', '2004', '2007', '2003', '1976', '1977', '1920', '1959', '1998']
YearRemodAdd (numeric, 61 distinct): ['1950', '2006', '2007', '2005', '2004', '2000', '2003', '2002', '2008', '1996']
RoofStyle (nominal, 6 distinct): ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed']
RoofMatl (nominal, 8 distinct): ['CompShg', 'Tar&Grv', 'WdShngl', 'WdShake', 'ClyTile', 'Membran', 'Metal', 'Roll']
Exterior1st (nominal, 15 distinct): ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood', 'CemntBd', 'BrkFace', 'WdShing', 'Stucco', 'AsbShng']
Exterior2nd (nominal, 16 distinct): ['VinylSd', 'MetalSd', 'HdBoard', 'Wd Sdng', 'Plywood', 'CmentBd', 'Wd Shng', 'Stucco', 'BrkFace', 'AsbShng']
MasVnrType (nominal, 5 distinct): ['None', 'BrkFace', 'Stone', 'BrkCmn']
MasVnrArea (numeric, 328 distinct): ['0.0', '180.0', '72.0', '108.0', '120.0', '16.0', '200.0', '340.0', '106.0', '80.0']
ExterQual (nominal, 4 distinct): ['TA', 'Gd', 'Ex', 'Fa']
ExterCond (nominal, 5 distinct): ['TA', 'Gd', 'Fa', 'Ex', 'Po']
Foundation (nominal, 6 distinct): ['PConc', 'CBlock', 'BrkTil', 'Slab', 'Stone', 'Wood']
BsmtQual (nominal, 5 distinct): ['TA', 'Gd', 'Ex', 'Fa']
BsmtCond (nominal, 5 distinct): ['TA', 'Gd', 'Fa', 'Po']
BsmtExposure (nominal, 5 distinct): ['No', 'Av', 'Gd', 'Mn']
BsmtFinType1 (nominal, 7 distinct): ['Unf', 'GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ']
BsmtFinSF1 (numeric, 637 distinct): ['0', '24', '16', '686', '662', '20', '936', '616', '560', '553']
BsmtFinType2 (nominal, 7 distinct): ['Unf', 'Rec', 'LwQ', 'BLQ', 'ALQ', 'GLQ']
BsmtFinSF2 (numeric, 144 distinct): ['0', '180', '374', '551', '147', '294', '391', '539', '96', '480']
BsmtUnfSF (numeric, 780 distinct): ['0', '728', '384', '600', '300', '572', '270', '625', '672', '440']
TotalBsmtSF (numeric, 721 distinct): ['0', '864', '672', '912', '1040', '816', '768', '728', '894', '780']
Heating (nominal, 6 distinct): ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']
HeatingQC (nominal, 5 distinct): ['Ex', 'TA', 'Gd', 'Fa', 'Po']
CentralAir (nominal, 2 distinct): ['Y', 'N']
Electrical (nominal, 6 distinct): ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
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
KitchenQual (nominal, 4 distinct): ['TA', 'Gd', 'Ex', 'Fa']
TotRmsAbvGrd (numeric, 12 distinct): ['6', '7', '5', '8', '4', '9', '10', '11', '3', '12']
Functional (nominal, 7 distinct): ['Typ', 'Min2', 'Min1', 'Mod', 'Maj1', 'Maj2', 'Sev']
Fireplaces (numeric, 4 distinct): ['0', '1', '2', '3']
FireplaceQu (nominal, 6 distinct): ['Gd', 'TA', 'Fa', 'Ex', 'Po']
GarageType (nominal, 7 distinct): ['Attchd', 'Detchd', 'BuiltIn', 'Basment', 'CarPort', '2Types']
GarageYrBlt (numeric, 98 distinct): ['2005.0', '2006.0', '2004.0', '2003.0', '2007.0', '1977.0', '1998.0', '1999.0', '1976.0', '2008.0']
GarageFinish (nominal, 4 distinct): ['Unf', 'RFn', 'Fin']
GarageCars (numeric, 5 distinct): ['2', '1', '3', '0', '4']
GarageArea (numeric, 441 distinct): ['0', '440', '576', '240', '484', '528', '288', '400', '264', '480']
GarageQual (nominal, 6 distinct): ['TA', 'Fa', 'Gd', 'Ex', 'Po']
GarageCond (nominal, 6 distinct): ['TA', 'Fa', 'Gd', 'Po', 'Ex']
PavedDrive (nominal, 3 distinct): ['Y', 'N', 'P']
WoodDeckSF (numeric, 274 distinct): ['0', '192', '100', '144', '120', '168', '140', '224', '208', '240']
OpenPorchSF (numeric, 202 distinct): ['0', '36', '48', '20', '40', '45', '24', '30', '60', '39']
EnclosedPorch (numeric, 120 distinct): ['0', '112', '96', '192', '144', '120', '216', '156', '116', '252']
3SsnPorch (numeric, 20 distinct): ['0', '168', '144', '180', '216', '290', '153', '96', '23', '162']
ScreenPorch (numeric, 76 distinct): ['0', '192', '120', '224', '189', '180', '147', '90', '160', '144']
PoolArea (numeric, 8 distinct): ['0', '512', '648', '576', '555', '480', '519', '738']
PoolQC (nominal, 4 distinct): ['Gd', 'Ex', 'Fa']
Fence (nominal, 5 distinct): ['MnPrv', 'GdPrv', 'GdWo', 'MnWw']
MiscFeature (nominal, 5 distinct): ['Shed', 'Gar2', 'Othr', 'TenC']
MiscVal (numeric, 21 distinct): ['0', '400', '500', '700', '450', '600', '2000', '1200', '480', '15500']
MoSold (numeric, 12 distinct): ['6', '7', '5', '4', '8', '3', '10', '11', '9', '12']
YrSold (numeric, 5 distinct): ['2009', '2007', '2006', '2008', '2010']
SaleType (nominal, 9 distinct): ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'ConLw', 'CWD', 'Oth', 'Con']
SaleCondition (nominal, 6 distinct): ['Normal', 'Partial', 'Abnorml', 'Family', 'Alloca', 'AdjLand']
'''

CONTEXT = "Prices of Boston Houses"
TARGET = CuratedTarget(raw_name='SalePrice', new_name="Boston House Price Median Value in thousands of USD",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []