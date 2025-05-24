from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: auction_verification
====
Examples: 2043
====
URL: https://www.openml.org/search?type=data&id=44958
====
Description: **Data Description**

This dataset was created to verify properties of an Simultaneous Multi-Round (SMR) auction model.
The creators of the dataset use BPMN to model the design of the German 4G spectrum auction to sell 800 MHz band. The auction has four bidders and six products. A random budget is assigned from the range [1, 100] to each bidder for each product. A reserve price of 3 is also defined for all products. Further, each bidder has an individual capacity.

Each instance in the dataset represents a simulation of an auction.

**Attribute Description**

1. *process.b1.capacity* - an integer in [0, 3], denoting the current capacities of the bidders
2. *process.b2.capacity* - an integer in [0, 3], denoting the current capacities of the bidders
3. *process.b3.capacity* - an integer in [0, 3], denoting the current capacities of the bidders
4. *process.b4.capacity* - an integer in [0, 3], denoting the current capacities of the bidders
5. *property.price* - an integer in [59, 90], denoting the price that is currently verified for the property.product
6. *property.product* - an integer in [1, 6], denoting the currently verified product
7. *property.winner* - an integer in [1, 4], denoting the bidder that is currently verified as winner for the property.product with the property.price. This feature is empty for iterations where the price is not clear yet.
8. *verification.result* - a boolean denoting if current property is satisfied in the underlying Petri Net or not, ignored column
9. *verification.time* - a positive integer, denoting the time (in ms) for verifying the current property against the underlying Petri Net, target feature
====
Target Variable: verification.time (numeric, 2039 distinct): ['757.9167', '1128.8333', '442.75', '717.5', '24781.75', '8954.6562', '84.3958', '35755.0', '1673.5', '1158.8333']
====
Features:

process.b1.capacity (numeric, 3 distinct): ['0', '1', '2']
process.b2.capacity (numeric, 4 distinct): ['2', '3', '1', '0']
process.b3.capacity (numeric, 2 distinct): ['2', '1']
process.b4.capacity (numeric, 2 distinct): ['1', '0']
property.price (numeric, 32 distinct): ['70', '80', '60', '69', '59', '61', '62', '63', '64', '65']
property.product (nominal, 6 distinct): ['2', '1', '6', '4', '5', '3']
property.winner (nominal, 5 distinct): ['0', '3', '2', '4', '1']
'''

CONTEXT = "Simultanous Multi-Round Auction Verification"
TARGET = CuratedTarget(raw_name="verification.time", new_name="Verification Time", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []