from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: spambase
====
Examples: 4601
====
URL: https://www.openml.org/search?type=data&id=44
====
Description: **Author**: Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt    
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/spambase)   
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

SPAM E-mail Database  
The "spam" concept is diverse: advertisements for products/websites, make money fast schemes, chain letters, pornography... Our collection of spam e-mails came from our postmaster and individuals who had filed spam.  Our collection of non-spam e-mails came from filed work and personal e-mails, and hence the word 'george' and the area code '650' are indicators of non-spam.  These are useful when constructing a personalized spam filter.  One would either have to blind such non-spam indicators or get a very wide collection of non-spam to generate a general purpose spam filter.
 
For background on spam:  
Cranor, Lorrie F., LaMacchia, Brian A.  Spam! Communications of the ACM, 41(8):74-83, 1998.  

### Attribute Information:  
The last column denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. Most of the attributes indicate whether a particular word or character was frequently occurring in the e-mail. The run-length attributes (55-57) measure the length of sequences of consecutive capital letters.  

For the statistical measures of each attribute, see the end of this file. Here are the definitions of the attributes:  

48 continuous real [0,100] attributes of type  
word_freq_WORD = percentage of words in the e-mail that match WORD,  i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail.  A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.
 
6 continuous real [0,100] attributes of type char_freq_CHAR = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail
 
1 continuous real [1,...] attribute of type capital_run_length_average
 = average length of uninterrupted sequences of capital letters
 
1 continuous integer [1,...] attribute of type capital_run_length_longest
 = length of longest uninterrupted sequence of capital letters
 
1 continuous integer [1,...] attribute of type capital_run_length_total
 = sum of length of uninterrupted sequences of capital letters
 = total number of capital letters in the e-mail
 
1 nominal {0,1} class attribute of type spam
 = denotes whether the e-mail was considered spam (1) or not (0), 
 i.e. unsolicited commercial e-mail.
====
Target Variable: class (nominal, 2 distinct): ['0', '1']
====
Features:

word_freq_make (numeric, 142 distinct): ['0.0', '0.1', '0.09', '0.17', '0.08', '0.05', '0.07', '0.06', '0.34', '0.33']
word_freq_address (numeric, 171 distinct): ['0.0', '14.28', '0.08', '0.1', '0.17', '0.19', '0.2', '0.26', '0.49', '0.39']
word_freq_all (numeric, 214 distinct): ['0.0', '0.32', '0.29', '0.55', '0.36', '0.4', '0.59', '0.71', '0.1', '0.27']
word_freq_3d (numeric, 43 distinct): ['0.0', '0.58', '0.42', '0.17', '0.21', '35.46', '0.57', '0.44', '7.07', '1.33']
word_freq_our (numeric, 255 distinct): ['0.0', '0.36', '0.32', '0.19', '0.8', '0.26', '0.29', '0.14', '0.68', '0.64']
word_freq_over (numeric, 141 distinct): ['0.0', '0.09', '0.1', '0.19', '0.03', '0.08', '0.16', '0.11', '0.32', '0.13']
word_freq_remove (numeric, 173 distinct): ['0.0', '0.08', '0.05', '0.5', '0.32', '0.19', '0.25', '0.1', '0.16', '0.4']
word_freq_internet (numeric, 170 distinct): ['0.0', '0.05', '0.18', '0.32', '0.1', '0.17', '0.33', '0.16', '0.08', '0.26']
word_freq_order (numeric, 144 distinct): ['0.0', '0.09', '0.08', '0.8', '0.1', '0.23', '0.16', '0.24', '0.05', '0.06']
word_freq_mail (numeric, 245 distinct): ['0.0', '0.08', '0.39', '0.1', '0.29', '0.27', '0.17', '0.35', '0.19', '0.23']
word_freq_receive (numeric, 113 distinct): ['0.0', '0.1', '0.26', '0.17', '0.3', '0.29', '0.14', '0.08', '0.31', '0.11']
word_freq_will (numeric, 316 distinct): ['0.0', '0.64', '0.33', '0.32', '0.55', '0.72', '0.54', '0.5', '0.34', '0.7']
word_freq_people (numeric, 158 distinct): ['0.0', '0.17', '0.19', '0.3', '0.32', '0.27', '0.12', '0.25', '0.29', '0.2']
word_freq_report (numeric, 133 distinct): ['0.0', '0.36', '0.05', '0.08', '0.17', '0.07', '0.06', '0.19', '1.19', '0.11']
word_freq_addresses (numeric, 118 distinct): ['0.0', '0.03', '1.61', '0.18', '0.16', '0.17', '0.05', '1.15', '0.02', '2.21']
word_freq_free (numeric, 253 distinct): ['0.0', '0.1', '0.32', '0.25', '0.23', '0.38', '0.19', '0.14', '0.08', '0.58']
word_freq_business (numeric, 197 distinct): ['0.0', '0.08', '0.32', '0.37', '0.19', '0.1', '0.2', '0.17', '0.7', '0.44']
word_freq_email (numeric, 229 distinct): ['0.0', '1.11', '0.08', '0.05', '0.32', '0.44', '0.06', '0.33', '0.12', '0.19']
word_freq_you (numeric, 575 distinct): ['0.0', '1.31', '2.0', '2.56', '3.33', '1.29', '3.84', '1.2', '1.36', '1.85']
word_freq_credit (numeric, 148 distinct): ['0.0', '0.17', '0.2', '0.14', '0.23', '0.16', '0.24', '0.06', '0.09', '0.39']
word_freq_your (numeric, 401 distinct): ['0.0', '1.36', '0.42', '0.64', '0.7', '1.23', '1.16', '1.35', '1.08', '1.25']
word_freq_font (numeric, 99 distinct): ['0.0', '0.17', '0.62', '8.33', '1.61', '0.84', '10.38', '0.2', '0.21', '8.29']
word_freq_000 (numeric, 164 distinct): ['0.0', '0.34', '0.36', '0.08', '0.6', '0.48', '0.85', '0.09', '0.39', '0.15']
word_freq_money (numeric, 143 distinct): ['0.0', '0.08', '0.32', '0.3', '0.34', '0.09', '0.1', '0.2', '0.05', '0.38']
word_freq_hp (numeric, 395 distinct): ['0.0', '0.49', '0.34', '1.58', '0.64', '2.22', '0.9', '1.78', '2.63', '0.44']
word_freq_hpl (numeric, 281 distinct): ['0.0', '0.74', '0.68', '1.19', '0.69', '0.58', '0.64', '0.42', '0.57', '0.26']
word_freq_george (numeric, 240 distinct): ['0.0', '20.0', '25.0', '0.05', '0.7', '0.08', '16.66', '2.0', '4.76', '4.34']
word_freq_650 (numeric, 200 distinct): ['0.0', '0.24', '4.76', '2.04', '0.68', '0.54', '0.66', '0.15', '0.63', '0.58']
word_freq_lab (numeric, 156 distinct): ['0.0', '0.5', '4.76', '0.58', '0.68', '0.39', '0.02', '0.54', '0.32', '0.93']
word_freq_labs (numeric, 179 distinct): ['0.0', '0.24', '0.17', '0.86', '0.27', '0.68', '0.18', '0.25', '0.52', '0.39']
word_freq_telnet (numeric, 128 distinct): ['0.0', '0.7', '0.58', '0.24', '0.27', '4.76', '0.26', '0.34', '0.22', '0.15']
word_freq_857 (numeric, 106 distinct): ['0.0', '0.58', '4.76', '0.39', '0.68', '0.17', '0.35', '0.55', '0.15', '0.24']
word_freq_data (numeric, 184 distinct): ['0.0', '0.34', '0.14', '0.08', '0.47', '0.33', '0.27', '0.23', '0.26', '0.16']
word_freq_415 (numeric, 110 distinct): ['0.0', '0.58', '4.76', '0.63', '0.17', '0.68', '0.39', '0.55', '0.15', '0.76']
word_freq_85 (numeric, 177 distinct): ['0.0', '0.1', '0.24', '0.33', '0.58', '0.5', '0.26', '0.34', '0.19', '0.29']
word_freq_technology (numeric, 159 distinct): ['0.0', '0.09', '0.43', '0.08', '0.58', '0.42', '0.13', '0.16', '0.34', '0.15']
word_freq_1999 (numeric, 188 distinct): ['0.0', '0.08', '0.24', '0.19', '0.64', '0.31', '0.23', '0.68', '0.1', '0.29']
word_freq_parts (numeric, 53 distinct): ['0.0', '0.29', '0.02', '0.11', '0.07', '0.03', '0.1', '0.12', '0.14', '0.09']
word_freq_pm (numeric, 163 distinct): ['0.0', '0.1', '0.09', '0.64', '0.19', '0.11', '0.31', '0.66', '0.58', '0.16']
word_freq_direct (numeric, 125 distinct): ['0.0', '0.08', '0.16', '0.17', '0.11', '0.46', '0.09', '0.12', '0.15', '0.1']
word_freq_cs (numeric, 108 distinct): ['0.0', '0.31', '7.14', '0.34', '1.44', '4.75', '0.1', '0.25', '0.28', '0.68']
word_freq_meeting (numeric, 186 distinct): ['0.0', '0.11', '0.06', '3.84', '0.08', '0.8', '0.03', '0.71', '0.28', '0.9']
word_freq_original (numeric, 136 distinct): ['0.0', '0.2', '0.17', '0.06', '0.24', '0.18', '0.02', '0.68', '0.11', '0.09']
word_freq_project (numeric, 160 distinct): ['0.0', '0.08', '0.06', '0.28', '0.05', '0.1', '0.16', '0.33', '0.02', '0.8']
word_freq_re (numeric, 230 distinct): ['0.0', '0.08', '0.1', '0.06', '0.33', '0.05', '0.12', '0.32', '0.27', '0.64']
word_freq_edu (numeric, 227 distinct): ['0.0', '0.08', '0.1', '0.09', '0.27', '0.34', '0.16', '0.33', '0.28', '0.8']
word_freq_table (numeric, 38 distinct): ['0.0', '0.04', '0.02', '0.03', '0.05', '0.19', '0.09', '0.01', '0.39', '0.16']
word_freq_conference (numeric, 106 distinct): ['0.0', '0.13', '0.2', '0.24', '0.1', '0.19', '0.15', '0.28', '0.14', '0.11']
char_freq_%3B (numeric, 313 distinct): ['0.0', '0.01', '0.019', '0.027', '0.015', '0.012', '0.011', '0.016', '0.014', '0.063']
char_freq_%28 (numeric, 641 distinct): ['0.0', '0.143', '0.052', '0.037', '0.11', '0.085', '0.105', '0.047', '0.058', '0.111']
char_freq_%5B (numeric, 225 distinct): ['0.0', '0.066', '0.031', '0.053', '0.028', '0.061', '0.047', '0.194', '0.03', '0.264']
char_freq_%21 (numeric, 964 distinct): ['0.0', '0.01', '0.149', '0.102', '0.055', '0.238', '0.045', '0.016', '0.082', '0.056']
char_freq_%24 (numeric, 504 distinct): ['0.0', '0.118', '0.061', '0.031', '0.158', '0.014', '0.062', '0.056', '0.107', '0.157']
char_freq_%23 (numeric, 316 distinct): ['0.0', '0.015', '0.054', '0.013', '0.031', '0.026', '0.012', '0.052', '0.033', '0.03']
capital_run_length_average (numeric, 2161 distinct): ['1.0', '2.0', '1.8', '1.5', '1.666', '1.25', '2.333', '1.333', '3.0', '3.333']
capital_run_length_longest (numeric, 271 distinct): ['1', '5', '11', '4', '12', '3', '7', '15', '13', '10']
capital_run_length_total (numeric, 919 distinct): ['5', '9', '7', '6', '4', '8', '53', '10', '11', '13']
'''

CONTEXT = "Spam Email Database"
TARGET = CuratedTarget(raw_name="class", new_name="Is Spam", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []