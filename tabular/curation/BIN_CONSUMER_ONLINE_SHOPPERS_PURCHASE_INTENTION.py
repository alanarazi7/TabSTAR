from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: online-shoppers-intention
====
Examples: 12330
====
URL: https://www.openml.org/search?type=data&id=45560
====
Description: ## Source:

1. C. Okan Sakar
Department of Computer Engineering, Faculty of
Engineering and Natural Sciences, Bahcesehir University,
34349 Besiktas, Istanbul, Turkey

2. Yomi Kastro
Inveon Information Technologies Consultancy and Trade,
34335 Istanbul, Turkey

## Data Set Information:

The dataset consists of feature vectors belonging to 12,330 sessions.
The dataset was formed so that each session
would belong to a different user in a 1-year period to avoid
any tendency to a specific campaign, special day, user
profile, or period.


## Attribute Information:

The dataset consists of 10 numerical and 8 categorical attributes.
The 'Revenue' attribute can be used as the class label.

"Administrative", "Administrative Duration", "Informational", "Informational Duration", "Product Related" and "Product Related Duration" represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories. The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another. The "Bounce Rate", "Exit Rate" and "Page Value" features represent the metrics measured by "Google Analytics" for each page in the e-commerce site. The value of "Bounce Rate" feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session. The value of "Exit Rate" feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session. The "Page Value" feature represents the average value for a web page that a user visited before completing an e-commerce transaction. The "Special Day" feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother's Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction. The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date. For example, for Valentina's day, this value takes a nonzero value between February 2 and February 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on February 8. The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.

## Relevant Papers:

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link]

## Note

* Compared to v1 this one contains correct variable coding.
====
Target Variable: Revenue (nominal, 2 distinct): ['0', '1']
====
Features:

Administrative (numeric, 27 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Administrative_Duration (numeric, 3335 distinct): ['0.0', '4.0', '5.0', '7.0', '11.0', '6.0', '14.0', '9.0', '15.0', '10.0']
Informational (numeric, 17 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
Informational_Duration (numeric, 1258 distinct): ['0.0', '9.0', '7.0', '10.0', '6.0', '12.0', '13.0', '16.0', '8.0', '11.0']
ProductRelated (numeric, 311 distinct): ['1', '2', '3', '4', '6', '7', '5', '8', '10', '9']
ProductRelated_Duration (numeric, 9551 distinct): ['0.0', '17.0', '11.0', '8.0', '15.0', '12.0', '19.0', '22.0', '13.0', '7.0']
BounceRates (numeric, 1872 distinct): ['0.0', '0.2', '0.0667', '0.0286', '0.05', '0.0333', '0.025', '0.0167', '0.1', '0.04']
ExitRates (numeric, 4777 distinct): ['0.2', '0.1', '0.05', '0.0333', '0.0667', '0.025', '0.04', '0.0167', '0.02', '0.0222']
PageValues (numeric, 2704 distinct): ['0.0', '53.988', '42.2931', '59.988', '16.1586', '44.8935', '14.1274', '34.04', '10.999', '58.9242']
SpecialDay (numeric, 6 distinct): ['0.0', '0.6', '0.8', '0.4', '0.2', '1.0']
Month (nominal, 10 distinct): ['May', 'Nov', 'Mar', 'Dec', 'Oct', 'Sep', 'Aug', 'Jul', 'June', 'Feb']
OperatingSystems (nominal, 8 distinct): ['2', '1', '3', '4', '8', '6', '7', '5']
Browser (nominal, 13 distinct): ['2', '1', '4', '5', '6', '10', '8', '3', '13', '7']
Region (nominal, 9 distinct): ['1', '3', '4', '2', '6', '7', '9', '8', '5']
TrafficType (nominal, 20 distinct): ['2', '1', '3', '4', '13', '10', '6', '8', '5', '11']
VisitorType (nominal, 3 distinct): ['Returning_Visitor', 'New_Visitor', 'Other']
Weekend (nominal, 2 distinct): ['0', '1']
'''

CONTEXT = "Online Shoppers Purchase Intention Prediction"
TARGET = CuratedTarget(raw_name="Revenue", new_name="Created Revenue", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []
