from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: REG_SOCIAL_BOOKS_GOODREADS
====
Examples: 3967
====
URL: http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv
====
Description: 
Goodreads: Datasets containing information about
books. The task is to predict the average rating of each
book.

====
Features:

ID (int64, 3967 distinct): ['0', '2635', '2637', '2638', '2639', '2640', '2641', '2642', '2643', '2644']
Title (object, 3455 distinct): ['Autobiography', 'An Autobiography', 'The Autobiography', 'My Autobiography', 'Autobiographies', 'The Way The Wind Blows: An Autobiography', 'The Test: My Autobiography', 'Cheech & Chong: The Unauthorized Autobiography', 'Autobiography of Allen Jay', 'James Toseland: The Autobiography']
Description (object, 2614 distinct): [' ', 'This book was converted from its physical edition to the digital format by a community of volunteers. You may find it for free on the web. Purchase of the Kindle edition includes wireless delivery.', 'Kessinger Publishing is the place to find hundreds of thousands of rare and hard-to-find books with something of interest for everyone', 'This is a reproduction of a book published before 1923. This book may have occasional imperfections such as missing or blurred pages, poor pictures, errant marks, etc. that were either part of the original artifact, or were introduced by the scanning process. We believe this work is culturally important, and despite the imperfections, have elected to bring it back into print as part of our continuing commitment to the preservation of printed works worldwide. We appreciate your understanding of the imperfections in the preservation process, and hope you enjoy this valuable book.', 'Many of the earliest books, particularly those dating back to the 1900s and before, are now extremely scarce and increasingly expensive. We are republishing these classic works in affordable, high quality, modern editions, using the original text and artwork.', 'This is a pre-1923 historical reproduction that was curated for quality. Quality assurance was conducted on each of these books in an attempt to remove books with imperfections introduced by the digitization process. Though we have made best efforts - the books may have occasional errors that do not impede the reading experience. We believe this work is culturally important and have elected to bring the book back into print as part of our continuing commitment to the preservation of printed works worldwide.', "This is an EXACT reproduction of a book published before 1923. This IS NOT an OCR'd book with strange characters, introduced typographical errors, and jumbled words. This book may have occasional imperfections such as missing or blurred pages, poor pictures, errant marks, etc. that were either part of the original artifact, or were introduced by the scanning process. We believe this work is culturally important, and despite the imperfections, have elected to bring it back into print as part of our continuing commitment to the preservation of printed works worldwide. We appreciate your understanding of the imperfections in the preservation process, and hope you enjoy this valuable book.", 'This work has been selected by scholars as being culturally important, and is part of the knowledge base of civilization as we know it. This work was reproduced from the original artifact, and remains as true to the original work as possible. Therefore, you will see the original copyright references, library stamps (as most of these works have been housed in our most important libraries around the world), and other notations in the work. This work is in the public domain in the United States of America, and possibly other nations. Within the United States, you may freely copy and distribute this work, as no entity (individual or corporate) has a copyright on the body of the work.As a reproduction of a historical artifact, this work may contain missing or blurred pages, poor pictures, errant marks, etc. Scholars believe, and we concur, that this work is important enough to be preserved, reproduced, and made generally available to the public. We appreciate your support of the preservation process, and thank you for being an important part of keeping this knowledge alive and relevant.', "This scarce antiquarian book is a selection from Kessinger Publishing's Legacy Reprint Series. Due to its age, it may contain imperfections such as marks, notations, marginalia and flawed pages. Because we believe this work is culturally important, we have made it available as part of our commitment to protecting, preserving, and promoting the world's literature. Kessinger Publishing is the place to find hundreds of thousands of rare and hard-to-find books with something of interest for everyone", "When Niall Quinn learned he was going to the 2002 World Cup with Ireland, it seemed the perfect climax to his international career. Yet even before the competition had started, Quinn was caught up in the most emotionally draining events of his career, as Ireland's World Cup campaign was rocked by Roy Keane's sudden departure. All his efforts at mediation failed, leaving him exhausted. As he worked to find a solution, Quinn looked back on his life and career, and saw echoes of his current situation. In this fascinating autobiography, updated for this edition, he recalls the all-night drinking sessions with Tony Adams and Paul Merson, the gambling, the good times and the bad. It is a remarkable story, brilliantly told."]
ISBN (object, 3080 distinct): ['0297792857', '386521472X', '0805463232', '1892446073', '1861761562', '0804606749', '0233997806', '0404200877', '0752888374', '0340708522']
ISBN13 (object, 3038 distinct): [' ', '9781477581728', '9780854300563', '9780882862118', '9780404200879', '9780752888378', '9780706438079', '9780805463231', '9781892446077', '9780340708521']
PageCount (int64, 576 distinct): ['0', '320', '256', '288', '224', '304', '192', '352', '336', '240']
FirstAuthor (object, 3211 distinct): ['Mark Twain', 'Benjamin Franklin', 'Booker T. Washington', 'Anonymous', 'Jon E. Lewis', "Seán O'Casey", 'Theodore Roosevelt', 'Thomas Jefferson', 'James Olney', 'Dr. Block']
SecondAuthor (object, 841 distinct): [' ', 'Benjamin Franklin', 'Julia   Watson', 'Tom Carter', 'Harriet E. Smith', 'Donald Day', 'Michael Simon', 'Robert H. Ferrell', 'Euan Cameron', 'Maureen Lipman']
ThirdAuthor (object, 190 distinct): [' ', 'Anita Pacheco', 'Ryan Giggs', 'David E. Schultz', 'Charles H. Red Corn', 'Mary Maher', 'Lily Chia Brissman', 'Eleanor Zelliot', 'Tomi Jill Folk', 'Billie Stafford']
Rating (float64, 214 distinct): ['4.0', '0.0', '3.0', '5.0', '3.5', '4.5', '3.67', '4.33', '3.75', '4.25']
NumberofRatings (int64, 473 distinct): ['1', '0', '2', '3', '4', '5', '6', '8', '7', '9']
NumberofReviews (object, 179 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Publisher (object, 1698 distinct): [' ', 'Hodder & Stoughton', 'Kessinger Publishing', 'Createspace', 'Orion Publishing', 'iUniverse', 'Simon & Schuster', 'Headline Book Publishing', 'Oxford University Pres', 'University of Wisconsin Press']
PublishDate (object, 1896 distinct): [' ', '2009', 'March 1st 2007', 'January 1st 1992', 'January 1st 2005', 'January 1st 1995', '1965', '2008', 'October 1st 2007', 'September 1st 2004']
Format (object, 35 distinct): ['Paperback', 'Hardcover', ' ', 'Kindle Edition', 'Unknown Binding', 'ebook', 'Nook', 'Library Binding', 'Mass Market Paperback', 'Audio CD']
Language (object, 15 distinct): ['English', ' ', 'German', 'French', 'Russian', 'Hungarian', 'Serbian', 'Swedish', 'Chinese', 'Slovenian']
FileName (object, 3967 distinct): ['100-1208987.Managing_My_Life.html', '3376-8609412-autobiography-of-an-actress.html', '3378-6236897-brushes-with-history.html', '3379-18732876-an-autobiography---the-original-classic-edition.html', '338-256827.Psycho.html', '3380-14293138-autobiography.html', '3381-2227014.The_Long_Road_Home.html', '3382-2191482.Hazel_O_Conner.html', '3383-2587879-the-autobiography-of-peggy-eaton.html', '3384-4389284-autobiographies.html']
'''

CONTEXT = "Books ratings"
TARGET = CuratedTarget(raw_name="Rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["FileName", "ID", ]
FEATURES = [CuratedFeature(raw_name="PublishDate", feat_type=FeatureType.DATE)]

DESCRIPTION = '''
Goodreads: Datasets containing information about
books. The task is to predict the average rating of each
book.
'''