from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: PhishingWebsites
====
Examples: 11055
====
URL: https://www.openml.org/search?type=data&id=4534
====
Description: **Author**: Rami Mustafa A Mohammad ( University of Huddersfield","rami.mohammad '@' hud.ac.uk","rami.mustafa.a '@' gmail.com) Lee McCluskey (University of Huddersfield","t.l.mccluskey '@' hud.ac.uk )  Fadi Thabtah (Canadian University of Dubai","fadi '@' cud.ac.ae)  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/phishing+websites)  
**Please cite**: Please refer to the [Machine Learning Repository's citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)  

Source:

Rami Mustafa A Mohammad ( University of Huddersfield, rami.mohammad '@' hud.ac.uk, rami.mustafa.a '@' gmail.com)
Lee McCluskey (University of Huddersfield,t.l.mccluskey '@' hud.ac.uk )
Fadi Thabtah (Canadian University of Dubai,fadi '@' cud.ac.ae)


Data Set Information:

One of the challenges faced by our research was the unavailability of reliable training datasets. In fact this challenge faces any researcher in the field. However, although plenty of articles about predicting phishing websites have been disseminated these days, no reliable training dataset has been published publically, may be because there is no agreement in literature on the definitive features that characterize phishing webpages, hence it is difficult to shape a dataset that covers all possible features. 
In this dataset, we shed light on the important features that have proved to be sound and effective in predicting phishing websites. In addition, we propose some new features.


Attribute Information:

For Further information about the features see the features file in the [data folder](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Phishing Websites Features.docx) of UCI.

Relevant Papers:

Mohammad, Rami, McCluskey, T.L. and Thabtah, Fadi (2012) An Assessment of Features Related to Phishing Websites using an Automated Technique. In: International Conferece For Internet Technology And Secured Transactions. ICITST 2012 . IEEE, London, UK, pp. 492-497. ISBN 978-1-4673-5325-0

Mohammad, Rami, Thabtah, Fadi Abdeljaber and McCluskey, T.L. (2014) Predicting phishing websites based on self-structuring neural network. Neural Computing and Applications, 25 (2). pp. 443-458. ISSN 0941-0643

Mohammad, Rami, McCluskey, T.L. and Thabtah, Fadi Abdeljaber (2014) Intelligent Rule based Phishing Websites Classification. IET Information Security, 8 (3). pp. 153-160. ISSN 1751-8709

 

Citation Request:

Please refer to the Machine Learning Repository's citation policy
====
Target Variable: Result (nominal, 2 distinct): ['1', '-1']
====
Features:

having_IP_Address (nominal, 2 distinct): ['1', '-1']
URL_Length (nominal, 3 distinct): ['-1', '1', '0']
Shortining_Service (nominal, 2 distinct): ['1', '-1']
having_At_Symbol (nominal, 2 distinct): ['1', '-1']
double_slash_redirecting (nominal, 2 distinct): ['1', '-1']
Prefix_Suffix (nominal, 2 distinct): ['-1', '1']
having_Sub_Domain (nominal, 3 distinct): ['1', '0', '-1']
SSLfinal_State (nominal, 3 distinct): ['1', '-1', '0']
Domain_registeration_length (nominal, 2 distinct): ['-1', '1']
Favicon (nominal, 2 distinct): ['1', '-1']
port (nominal, 2 distinct): ['1', '-1']
HTTPS_token (nominal, 2 distinct): ['1', '-1']
Request_URL (nominal, 2 distinct): ['1', '-1']
URL_of_Anchor (nominal, 3 distinct): ['0', '-1', '1']
Links_in_tags (nominal, 3 distinct): ['0', '-1', '1']
SFH (nominal, 3 distinct): ['-1', '1', '0']
Submitting_to_email (nominal, 2 distinct): ['1', '-1']
Abnormal_URL (nominal, 2 distinct): ['1', '-1']
Redirect (nominal, 2 distinct): ['0', '1']
on_mouseover (nominal, 2 distinct): ['1', '-1']
RightClick (nominal, 2 distinct): ['1', '-1']
popUpWidnow (nominal, 2 distinct): ['1', '-1']
Iframe (nominal, 2 distinct): ['1', '-1']
age_of_domain (nominal, 2 distinct): ['1', '-1']
DNSRecord (nominal, 2 distinct): ['1', '-1']
web_traffic (nominal, 3 distinct): ['1', '-1', '0']
Page_Rank (nominal, 2 distinct): ['-1', '1']
Google_Index (nominal, 2 distinct): ['1', '-1']
Links_pointing_to_page (nominal, 3 distinct): ['0', '1', '-1']
Statistical_report (nominal, 2 distinct): ['1', '-1']
'''

CONTEXT = "Phishing Detection Websites"
TARGET = CuratedTarget(raw_name="Result", new_name="Website Legitimacy", task_type=SupervisedTask.BINARY,
                       label_mapping={'1': "Phishing", '-1': "Legitimate"})
COLS_TO_DROP = []
FEATURES = []
