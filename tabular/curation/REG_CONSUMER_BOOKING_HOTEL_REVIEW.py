from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: 515K-Hotel-Reviews-Data-in-Europe
====
Examples: 515738
====
URL: https://www.openml.org/search?type=data&id=43712
====
Description: Acknowledgements
The data was scraped from Booking.com. All data in the file is publicly available to everyone already. Data is originally owned by Booking.com. Please contact me through my profile if you want to use this dataset somewhere else.
Data Context
This dataset contains 515,000 customer reviews and scoring of 1493 luxury hotels across Europe. Meanwhile, the geographical location of hotels are also provided for further analysis.
Data Content
The csv file contains 17 fields. The description of each field is as below:

Hotel_Address: Address of hotel. 
Review_Date: Date when reviewer posted the corresponding review.
Average_Score: Average Score of the hotel, calculated based on the latest comment in the last year.
Hotel_Name: Name of Hotel
Reviewer_Nationality: Nationality of Reviewer
Negative_Review: Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'
ReviewTotalNegativeWordCounts: Total number of words in the negative review.
Positive_Review: Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive'
ReviewTotalPositiveWordCounts: Total number of words in the positive review.
Reviewer_Score: Score the reviewer has given to the hotel, based on his/her experience
TotalNumberofReviewsReviewerHasGiven: Number of Reviews the reviewers has given in the past.
TotalNumberof_Reviews: Total number of valid reviews the hotel has.
Tags: Tags reviewer gave the hotel.
dayssincereview: Duration between the review date and scrape date.
AdditionalNumberof_Scoring: There are also some guests who just made a scoring on the service rather than a review. This number indicates how many valid scores without review in there.
lat: Latitude of the hotel
lng: longtitude of the hotel

In order to keep the text data clean, I removed unicode and punctuation in the text data and transform text into lower case. No other preprocessing was performed.
Inspiration
The dataset is large and informative, I believe you can have a lot of fun with it! Let me put some ideas below to futher inspire kagglers!

Fit a regression model on reviews and score to see which words are more indicative to a higher/lower score
Perform a sentiment analysis on the reviews
Find correlation between reviewer's nationality and scores.
Beautiful and informative visualization on the dataset.
Clustering hotels based on reviews
Simple recommendation engine to the guest who is fond of a special characteristic of hotel.

The idea is unlimited! Please, have a look into data, generate some ideas and leave a master kernel here! I am ready to upvote your ideas and kernels! Cheers!
====
Features:

Hotel_Address (string, 1493 distinct): ['163 Marsh Wall Docklands Tower Hamlets London E14 9SJ United Kingdom', '372 Strand Westminster Borough London WC2R 0JJ United Kingdom', 'Westminster Bridge Road Lambeth London SE1 7UT United Kingdom', 'Scarsdale Place Kensington Kensington and Chelsea London W8 5SY United Kingdom', '7 Pepys Street City of London London EC3N 4AF United Kingdom', '1 Inverness Terrace Westminster Borough London W2 3JP United Kingdom', 'Wrights Lane Kensington and Chelsea London W8 5SP United Kingdom', '225 Edgware Road Westminster Borough London W2 1JU United Kingdom', '4 18 Harrington Gardens Kensington and Chelsea London SW7 4LH United Kingdom', '1 Waterview Drive Greenwich London SE10 0TW United Kingdom']
Additional_Number_of_Scoring (numeric, 480 distinct): ['2682', '2288', '2623', '1831', '1936', '256', '1274', '832', '211', '404']
Review_Date (string, 731 distinct): ['8/2/2017', '9/15/2016', '4/5/2017', '8/30/2016', '2/16/2016', '7/5/2016', '5/31/2016', '12/5/2016', '7/12/2016', '8/2/2016']
Average_Score (numeric, 34 distinct): ['8.4', '8.1', '8.5', '8.7', '8.6', '8.2', '8.3', '8.8', '8.9', '8.0']
Hotel_Name (string, 1492 distinct): ['Britannia International Hotel Canary Wharf', 'Strand Palace Hotel', 'Park Plaza Westminster Bridge London', 'Copthorne Tara Hotel London Kensington', 'DoubleTree by Hilton Hotel London Tower of London', 'Grand Royale London Hyde Park', 'Holiday Inn London Kensington', 'Hilton London Metropole', 'Millennium Gloucester Hotel London', 'Intercontinental London The O2']
Reviewer_Nationality (string, 227 distinct): [' United Kingdom ', ' United States of America ', ' Australia ', ' Ireland ', ' United Arab Emirates ', ' Saudi Arabia ', ' Netherlands ', ' Switzerland ', ' Germany ', ' Canada ']
Negative_Review (string, 330011 distinct): ['No Negative', ' Nothing', ' Nothing ', ' nothing', ' N A', ' None', ' ', ' N a', ' Breakfast', ' Small room']
Review_Total_Negative_Word_Counts (numeric, 402 distinct): ['0', '2', '3', '6', '5', '7', '4', '8', '9', '10']
Total_Number_of_Reviews (numeric, 1142 distinct): ['9086', '9568', '12158', '7105', '7491', '6539', '5945', '6977', '5726', '4204']
Positive_Review (string, 412601 distinct): ['No Positive', ' Location', ' Everything', ' location', ' Nothing', ' The location', ' Great location', ' Good location', ' Location ', ' Everything ']
Review_Total_Positive_Word_Counts (numeric, 365 distinct): ['0', '6', '5', '4', '7', '8', '3', '9', '2', '10']
Total_Number_of_Reviews_Reviewer_Has_Given (numeric, 198 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Reviewer_Score (numeric, 37 distinct): ['10.0', '9.6', '9.2', '8.8', '8.3', '7.5', '7.9', '7.1', '6.7', '6.3']
Tags (string, 55242 distinct): ["[' Leisure trip ', ' Couple ', ' Double Room ', ' Stayed 1 night ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Standard Double Room ', ' Stayed 1 night ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Superior Double Room ', ' Stayed 1 night ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Deluxe Double Room ', ' Stayed 1 night ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Double Room ', ' Stayed 2 nights ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Superior Double Room ', ' Stayed 2 nights ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Standard Double Room ', ' Stayed 2 nights ', ' Submitted from a mobile device ']", "[' Leisure trip ', ' Couple ', ' Double Room ', ' Stayed 1 night ']", "[' Leisure trip ', ' Couple ', ' Standard Double Room ', ' Stayed 1 night ']", "[' Leisure trip ', ' Couple ', ' Deluxe Double Room ', ' Stayed 2 nights ', ' Submitted from a mobile device ']"]
days_since_review (string, 731 distinct): ['1 days', '322 day', '120 day', '338 day', '534 day', '394 day', '429 day', '241 day', '387 day', '366 day']
lat (numeric, 1473 distinct): ['51.5019', '51.5111', '51.501', '51.499', '51.5108', '51.511', '51.5', '51.5196', '51.4935', '51.5024']
lng (numeric, 1473 distinct): ['-0.0232', '-0.1209', '-0.1166', '-0.1917', '-0.0781', '-0.1863', '-0.1929', '-0.1705', '-0.1834', '-0.0002']
'''

CONTEXT = "Booking.com Hotel Reviews in Europe"
TARGET = CuratedTarget(raw_name="Reviewer_Score", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Review_Date", feat_type=FeatureType.DATE)]
