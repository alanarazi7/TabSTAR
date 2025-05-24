from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Titanic
====
URL: https://www.openml.org/search?type=data&id=40945
====
Description: **Author**: Frank E. Harrell Jr., Thomas Cason  
**Source**: [Vanderbilt Biostatistics](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.html)  
**Please cite**:   

The original Titanic dataset, describing the survival status of individual passengers on the Titanic. The titanic data does not contain information from the crew, but it does contain actual ages of half of the passengers. The principal source for data about Titanic passengers is the Encyclopedia Titanica. The datasets used here were begun by a variety of researchers. One of the original sources is Eaton & Haas (1994) Titanic: Triumph and Tragedy, Patrick Stephens Ltd, which includes a passenger list created by many researchers and edited by Michael A. Findlay.

Thomas Cason of UVa has greatly updated and improved the titanic data frame using the Encyclopedia Titanica and created the dataset here. Some duplicate passengers have been dropped, many errors corrected, many missing ages filled in, and new variables created. 

For more information about how this dataset was constructed:
http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3info.txt


### Attribute information  

The variables on our extracted dataset are pclass, survived, name, age, embarked, home.dest, room, ticket, boat, and sex. pclass refers to passenger class (1st, 2nd, 3rd), and is a proxy for socio-economic class. Age is in years, and some infants had fractional values. The titanic2 data frame has no missing data and includes records for the crew, but age is dichotomized at adult vs. child. These data were obtained from Robert Dawson, Saint Mary's University, E-mail. The variables are pclass, age, sex, survived. These data frames are useful for demonstrating many of the functions in Hmisc as well as demonstrating binary logistic regression analysis using the Design library. For more details and references see Simonoff, Jeffrey S (1997): The "unusual episode" and a second statistics course. J Statistics Education, Vol. 5 No. 1.
====
Target Variable: survived (nominal, 2 distinct): ['0', '1']
====
Features:

pclass (numeric, 3 distinct): ['3', '1', '2']
name (string, 1307 distinct): ['Connolly, Miss. Kate', 'Kelly, Mr. James', 'Allen, Miss. Elisabeth Walton', 'Ilmakangas, Miss. Ida Livija', 'Ilieff, Mr. Ylio', 'Ibrahim Shawah, Mr. Yousseff', 'Hyman, Mr. Abraham', 'Humblen, Mr. Adolf Mathias Nicolai Olsen', 'Howard, Miss. May Elizabeth', 'Horgan, Mr. John']
sex (nominal, 2 distinct): ['male', 'female']
age (numeric, 361 distinct): ['24.0', '22.0', '21.0', '30.0', '18.0', '25.0', '28.0', '36.0', '26.0', '29.0']
sibsp (numeric, 7 distinct): ['0', '1', '2', '4', '3', '8', '5']
parch (numeric, 8 distinct): ['0', '1', '2', '3', '4', '5', '6', '9']
ticket (string, 929 distinct): ['CA. 2343', '1601', 'CA 2144', 'PC 17608', '347077', '347082', '3101295', 'S.O.C. 14879', '113781', '19950']
fare (numeric, 282 distinct): ['8.05', '13.0', '7.75', '26.0', '7.8958', '10.5', '7.775', '7.2292', '7.925', '26.55']
cabin (string, 187 distinct): ['C23 C25 C27', 'G6', 'B57 B59 B63 B66', 'F4', 'F33', 'B96 B98', 'D', 'C22 C26', 'F2', 'C78']
embarked (nominal, 4 distinct): ['S', 'C', 'Q']
boat (string, 28 distinct): ['13', 'C', '15', '14', '4', '10', '5', '3', '9', '11']
body (numeric, 1309 distinct): ['135.0', '101.0', '37.0', '285.0', '156.0', '143.0', '120.0', '306.0', '69.0', '188.0']
home.dest (string, 370 distinct): ['New York, NY', 'London', 'Montreal, PQ', 'Paris, France', 'Cornwall / Akron, OH', 'Wiltshire, England Niagara Falls, NY', 'Winnipeg, MB', 'Philadelphia, PA', 'Belfast', 'Sweden Winnipeg, MN']
'''

CONTEXT = "Titanic Survival"
TARGET = CuratedTarget(raw_name="survived", task_type=SupervisedTask.BINARY, label_mapping={'0': "No", '1': "Yes"})
COLS_TO_DROP = []
FEATURES = []
