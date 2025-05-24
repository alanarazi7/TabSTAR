from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: internet_usage
====
Examples: 10108
====
URL: https://www.openml.org/search?type=data&id=372
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Internet Usage Data
 
 Data Type
 
    multivariate
 
 Abstract
 
    This data contains general demographic information on internet users
    in 1997.
 
 Sources
 
     Original Owner
 
 [1]Graphics, Visualization, & Usability Center
 College of Computing
 Geogia Institute of Technology
 Atlanta, GA
 
     Donor
 
 [2]Dr Di Cook
 Department of Statistics
 Iowa State University
 
    Date Donated: June 30, 1999
 
 Data Characteristics
 
    This data comes from a survey conducted by the Graphics and
    Visualization Unit at Georgia Tech October 10 to November 16, 1997.
    The full details of the survey are available [3]here.
 
    The particular subset of the survey provided here is the "general
    demographics" of internet users. The data have been recoded as
    entirely numeric, with an index to the codes described in the "Coding"
    file.
 
    The full survey is available from the web site above, along with
    summaries, tables and graphs of their analyses. In addition there is
    information on other parts of the survey, including technology
    demographics and web commerce.
 
 Data Format
 
    The data is stored in an ASCII files with one observation per line.
    Spaces separate fields.
 
 Past Usage
 
    This data was used in the American Statistical Association Statistical
    Graphics and Computing Sections 1999 Data Exposition.
      _________________________________________________________________
 
 
     [4]The UCI KDD Archive
     [5]Information and Computer Science
     [6]University of California, Irvine
     Irvine, CA 92697-3425
 
    Last modified: June 30, 1999
 
 References
 
    1. http://www.gvu.gatech.edu/gvu/user_surveys/survey-1997-10/
    2. http://www.public.iastate.edu/~dicook/
    3. http://www.cc.gatech.edu/gvu/user_surveys/survey-1997-10/
    4. http://kdd.ics.uci.edu/
    5. http://www.ics.uci.edu/
    6. http://www.uci.edu/


 Information about the dataset
 CLASSTYPE: nominal
 CLASSINDEX: none specific
====
Target Variable: Actual_Time (nominal, 46 distinct): ['Other', 'College_Student', 'Retired', 'Manager', 'Programmer', 'Service_Industry_Occupation', 'K-12_Student', 'Homemaker', 'Administrator/Secretary', 'Engineer']
====
Features:

Age (nominal, 77 distinct): ['26', '27', '28', '25', 'Not_Say', '24', '22', '18', '23', '29']
Community_Building (nominal, 4 distinct): ['More', 'Dont_Know', 'Equally', 'Less']
Community_Membership_Family (nominal, 2 distinct): ['0', '1']
Community_Membership_Hobbies (nominal, 2 distinct): ['0', '1']
Community_Membership_None (nominal, 2 distinct): ['0', '1']
Community_Membership_Other (nominal, 2 distinct): ['0', '1']
Community_Membership_Political (nominal, 2 distinct): ['0', '1']
Community_Membership_Professional (nominal, 2 distinct): ['0', '1']
Community_Membership_Religious (nominal, 2 distinct): ['0', '1']
Community_Membership_Support (nominal, 2 distinct): ['0', '1']
Country (nominal, 129 distinct): ['California', 'New_York', 'Texas', 'Florida', 'Georgia', 'Pennsylvania', 'Ohio', 'Illinois', 'Michigan', 'Washington']
Disability_Cognitive (nominal, 2 distinct): ['0', '1']
Disability_Hearing (nominal, 2 distinct): ['0', '1']
Disability_Motor (nominal, 2 distinct): ['0', '1']
Disability_Not_Impaired (nominal, 2 distinct): ['1', '0']
Disability_Not_Say (nominal, 2 distinct): ['0', '1']
Disability_Vision (nominal, 2 distinct): ['0', '1']
Education_Attainment (nominal, 9 distinct): ['Some_College', 'College', 'Masters', 'High_School', 'Special', 'Professional', 'Doctoral', 'Grammar', 'Other']
Falsification_of_Information (nominal, 7 distinct): ['Never', 'Under_25', 'NA', '26-50', 'Over_75', 'Not_Say', '51-75']
Gender (nominal, 2 distinct): ['Male', 'Female']
Household_Income (nominal, 9 distinct): ['$50-74', 'Not_Say', '$30-39', '$20-29', '$40-49', '$75-99', 'Over_$100', '$10-19', 'Under_$10']
How_You_Heard_About_Survey_Banner (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Friend (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Mailing_List (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Others (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Printed_Media (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Remebered (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Search_Engine (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_Usenet_News (nominal, 2 distinct): ['0', '1']
How_You_Heard_About_Survey_WWW_Page (nominal, 2 distinct): ['1', '0']
Major_Geographical_Location (nominal, 10 distinct): ['USA', 'Europe', 'Canada', 'Oceania', 'Asia', 'Africa', 'Middle_East', 'South_America', 'Central_America', 'West_Indies']
Major_Occupation (nominal, 5 distinct): ['Education', 'Other', 'Professional', 'Computer', 'Management']
Marital_Status (nominal, 7 distinct): ['Married', 'Single', 'Other', 'Divorced', 'Not_Say', 'Widowed', 'Separated']
Most_Import_Issue_Facing_the_Internet (nominal, 9 distinct): ['Privacy', 'Censorship', 'Navigation', 'Taxes', 'Other', 'Dont_Know', 'Encryption', 'Culture', 'Language']
Opinions_on_Censorship (nominal, 4 distinct): ['1', '4', '2', '3']
Primary_Computing_Platform (nominal, 12 distinct): ['Win95', 'Macintosh', 'Windows', 'NT', 'Unix', 'Dont_Know', 'OS2', 'PC_Unix', 'DOS', 'Other']
Primary_Language (nominal, 119 distinct): ['English', 'German', 'French', 'Dutch', 'Spanish', 'Chinese', 'Swedish', 'Italian', 'Norwegian', 'Not_Say']
Primary_Place_of_WWW_Access (nominal, 9 distinct): ['Home', 'Primarily_home', 'Primarily_work', 'Work', 'School', 'Other', 'Public', 'Distributed', 'Friend']
Race (nominal, 8 distinct): ['White', 'Asian', 'Other', 'Not_Say', 'Black', 'Hispanic', 'Latino', 'Indigenous']
Not_Purchasing_Bad_experience (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Bad_press (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Cant_find (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Company_policy (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Easier_locally (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Enough_info (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Judge_quality (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Never_tried (nominal, 2 distinct): ['0', '1']
Not_Purchasing_No_credit (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Not_applicable (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Not_option (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Other (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Prefer_people (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Privacy (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Receipt (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Security (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Too_complicated (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Uncomfortable (nominal, 2 distinct): ['0', '1']
Not_Purchasing_Unfamiliar_vendor (nominal, 2 distinct): ['0', '1']
Registered_to_Vote (nominal, 4 distinct): ['Yes', 'No', 'Not_Applicable', 'Not_Say']
Sexual_Preference (nominal, 6 distinct): ['Heterosexual', 'Not_say', 'Gay_male', 'Bisexual', 'Lesbian', 'Transgender']
Web_Ordering (nominal, 3 distinct): ['Yes', 'No', 'Dont_know']
Web_Page_Creation (nominal, 3 distinct): ['No', 'Yes', 'Dont_know']
Who_Pays_for_Access_Dont_Know (nominal, 2 distinct): ['0', '1']
Who_Pays_for_Access_Other (nominal, 2 distinct): ['0', '1']
Who_Pays_for_Access_Parents (nominal, 2 distinct): ['0', '1']
Who_Pays_for_Access_School (nominal, 2 distinct): ['0', '1']
Who_Pays_for_Access_Self (nominal, 2 distinct): ['1', '0']
Who_Pays_for_Access_Work (nominal, 2 distinct): ['0', '1']
Willingness_to_Pay_Fees (nominal, 8 distinct): ['Other_sources', 'Already_paying', 'Cost_too_high', 'Other', 'Poor_quality', 'Dont_trust', 'Payment_mechanism', 'Yes_regardless']
Years_on_Internet (nominal, 5 distinct): ['1-3_yr', 'Under_6_mo', '6-12_mo', '4-6_yr', 'Over_7_yr']
who (nominal, 10108 distinct): ['42', '96498', '96491', '96492', '96493', '96494', '96495', '96496', '96497', '96499']
'''

CONTEXT = "Ealy 1997 Internet Users Profession"
TARGET = CuratedTarget(raw_name="Actual_Time", new_name="Profession", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = []
