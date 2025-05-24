from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask, FeatureType

'''
Dataset Name: mental-health-in-tech-survey
====
Examples: 1259
====
URL: https://www.openml.org/search?type=data&id=46719
====
Description: About Dataset
Dataset Information
This dataset is from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders in the tech workplace. You are also encouraged to analyze data from the ongoing 2016 survey found here.

Content
This dataset contains the following data:

Timestamp

Age

Gender

Country

state: If you live in the United States, which state or territory do you live in?

self_employed: Are you self-employed?

family_history: Do you have a family history of mental illness?

treatment: Have you sought treatment for a mental health condition?

work_interfere: If you have a mental health condition, do you feel that it interferes with your work?

no_employees: How many employees does your company or organization have?

remote_work: Do you work remotely (outside of an office) at least 50% of the time?

tech_company (target for uploading to OpenML): Is your employer primarily a tech company/organization?

benefits: Does your employer provide mental health benefits?

care_options: Do you know the options for mental health care your employer provides?

wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?

seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?

anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?

leave: How easy is it for you to take medical leave for a mental health condition?

mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?

phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?

coworkers: Would you be willing to discuss a mental health issue with your coworkers?

supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?

mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?

phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?

mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?

obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?

comments: Any additional notes or comments

Inspiration
Some questions worth exploring:

How does the frequency of mental health illness and attitudes towards mental health vary by geographic location?
What are the strongest predictors of mental health illness or certain attitudes towards mental health in the workplace?
Acknowledgements
The original dataset is from Open Sourcing Mental Illness and can be downloaded here.
====
Target Variable: tech_company (string, 2 distinct): ['Yes', 'No']
====
Features:

Timestamp (string, 1246 distinct): ['2014-08-27 12:44:51', '2014-08-27 14:22:43', '2014-08-27 12:31:41', '2014-08-28 16:52:34', '2014-08-27 17:33:52', '2014-08-27 12:54:11', '2014-08-27 15:55:07', '2014-08-27 12:37:50', '2014-08-27 15:23:51', '2014-08-28 09:59:39']
Age (numeric, 53 distinct): ['29', '32', '26', '27', '33', '28', '31', '34', '30', '25']
Gender (string, 49 distinct): ['Male', 'male', 'Female', 'M', 'female', 'F', 'm', 'f', 'Make', 'Male ']
Country (string, 48 distinct): ['United States', 'United Kingdom', 'Canada', 'Germany', 'Ireland', 'Netherlands', 'Australia', 'France', 'India', 'New Zealand']
state (string, 45 distinct): ['CA', 'WA', 'NY', 'TN', 'TX', 'OH', 'IL', 'OR', 'PA', 'IN']
self_employed (string, 2 distinct): ['No', 'Yes']
family_history (string, 2 distinct): ['No', 'Yes']
treatment (string, 2 distinct): ['Yes', 'No']
work_interfere (string, 4 distinct): ['Sometimes', 'Never', 'Rarely', 'Often']
no_employees (string, 6 distinct): ['6-25', '26-100', 'More than 1000', '100-500', '1-5', '500-1000']
remote_work (string, 2 distinct): ['No', 'Yes']
benefits (string, 3 distinct): ['Yes', "Don't know", 'No']
care_options (string, 3 distinct): ['No', 'Yes', 'Not sure']
wellness_program (string, 3 distinct): ['No', 'Yes', "Don't know"]
seek_help (string, 3 distinct): ['No', "Don't know", 'Yes']
anonymity (string, 3 distinct): ["Don't know", 'Yes', 'No']
leave (string, 5 distinct): ["Don't know", 'Somewhat easy', 'Very easy', 'Somewhat difficult', 'Very difficult']
mental_health_consequence (string, 3 distinct): ['No', 'Maybe', 'Yes']
phys_health_consequence (string, 3 distinct): ['No', 'Maybe', 'Yes']
coworkers (string, 3 distinct): ['Some of them', 'No', 'Yes']
supervisor (string, 3 distinct): ['Yes', 'No', 'Some of them']
mental_health_interview (string, 3 distinct): ['No', 'Maybe', 'Yes']
phys_health_interview (string, 3 distinct): ['Maybe', 'No', 'Yes']
mental_vs_physical (string, 3 distinct): ["Don't know", 'Yes', 'No']
obs_consequence (string, 2 distinct): ['No', 'Yes']
comments (string, 160 distinct): ['* Small family business - YMMV.', "I'm not on my company's health insurance which could be part of the reason I answered Don't know to so many questions.", '(yes but the situation was unusual and involved a change in leadership at a very high level in the organization as well as an extended leave of absence)', 'None of us who are already in marginal groups in tech--the non-young the non-male the non-white--will risk our careers to admit another source of stigma: poor health.', "I have been incredibly public about my own struggle in my own conversations and in social media insofar as how I can use my depression to raise awareness or help others. Because of that my employer - or any future employer - kind of knows by default. It's not a secret. That said the downside of that openness is that I have no faith that I wouldn't be discriminated against at a future job simply because the information is public. Likewise I worry I'm seen as less-than by my employer in some circumstances. Regerdless I don't regret being public and raising awareness. My point is that even those of us who do publicly discuss the issue fear systemic retribution. ", 'At a previous employer I witness a bad thing happen to a coworker with mental health issues get swept under the rug... :(', 'While not personally affected I do have immediate family with mental health illness and my employer has been very supportive. Thanks for doing this survey.', "The company I work for was started by engineers and so anything other then the engineering department has always lacked a bit. Now that we've grown things are better but I feel that overall our total benefits package (including healthcare) isn't well communicated. This reflects negatively on the mental health questions above but would also reflect negatively on any other sort of survey about the benefits. That is I don't think the company is purposefully doing less for mental health. They just aren't doing enough across the board and that includes mental health.", 'Thank you for all you are doing to study this topic and raise awareness in our communities. ', "The main reason for the openness answers are because of an experience with my last employer. I felt I could trust my direct supervisor so I divulged information. It ended up spreading to more supervisors and eventually my coworkers. Supers highly suggested treatment but rushed things that shouldn't have been rushed and I ended up being incorrectly treated in a psych ward and mentally scarred from the issue. I lost most of my desire to program due to the experience not to mention thousands of dollars I lost - lost work time vacation time they used for treatment time doctors expenses etc. I have major depressive disorder high anxiety and mild agoraphobia. After seeing what treatment has to offer I will likely not seek it again and continue as is. (Long story short.)"]
'''

CONTEXT = "Mental Health in Tech Survey"
TARGET = CuratedTarget(raw_name="tech_company", new_name="Tech Company", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Timestamp", feat_type=FeatureType.DATE),]