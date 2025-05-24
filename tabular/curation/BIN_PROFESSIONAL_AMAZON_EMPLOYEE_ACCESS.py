from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: Amazon_employee_access
====
Examples: 32769
====
URL: https://www.openml.org/search?type=data&id=4135
====
Description: **Author**:   
**Source**: [Kaggle Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge)  
**Please cite**:   

### Description

The data consists of real historical data collected from 2010 & 2011.  Employees are manually allowed or denied access to resources over time. 
The data is used to create an algorithm capable of learning from this historical data to predict approval/denial for an unseen set of employees.

### Dataset Information

When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals. It is assumed that employees fulfilling the functions of a given role will access the same or similar resources. It is often the case that employees figure out the access they need as they encounter roadblocks during their daily work (e.g. not able to log into a reporting portal). A knowledgeable supervisor then takes time to manually grant the needed access in order to overcome access obstacles. As employees move throughout a company, this access discovery/recovery cycle wastes a non-trivial amount of time and money.

There is a considerable amount of data regarding an employee&rsquo;s role within an organization and the resources to which they have access. Given the data related to current employees and their provisioned access, models can be built that automatically determine access privileges as employees enter and leave roles within a company. These auto-access models seek to minimize the human involvement required to grant or revoke employee access.

The original training and test set were merged.

### Attributes Information

* ACTION [target]: ACTION is 1 if the resource was approved, 0 if the resource was not  
* RESOURCE: An ID for each resource  
* MGR_ID: The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time  
* ROLE_ROLLUP_1: Company role grouping category id 1 (e.g. US Engineering)  
* ROLE_ROLLUP_2: Company role grouping category id 2 (e.g. US Retail)  
* ROLE_DEPTNAME: Company role department description (e.g. Retail)  
* ROLE_TITLE: Company role business title description (e.g. Senior Engineering Retail Manager)  
* ROLE_FAMILY_DESC: Company role family extended description (e.g. Retail Manager, Software Engineering)  
* ROLE_FAMILY: Company role family description (e.g. Retail Manager)  
* ROLE_CODE: Company role code; this code is unique to each role (e.g. Manager)
====
Target Variable: target (nominal, 2 distinct): ['1', '0']
====
Features:

RESOURCE (nominal, 7518 distinct): ['4675', '79092', '25993', '75078', '3853', '75834', '6977', '32270', '42085', '17308']
MGR_ID (nominal, 4243 distinct): ['770', '2270', '2594', '1350', '2014', '16850', '7807', '3966', '3526', '5244']
ROLE_ROLLUP_1 (nominal, 128 distinct): ['117961', '117902', '91261', '118315', '118212', '118290', '119062', '118887', '117916', '118169']
ROLE_ROLLUP_2 (nominal, 177 distinct): ['118300', '118343', '118327', '118225', '118386', '118052', '117962', '118413', '118446', '118026']
ROLE_DEPTNAME (nominal, 449 distinct): ['117878', '117941', '117945', '118514', '117920', '117884', '119598', '118403', '119181', '120722']
ROLE_TITLE (nominal, 343 distinct): ['118321', '117905', '118784', '117879', '118568', '117885', '118054', '118685', '118777', '118451']
ROLE_FAMILY_DESC (nominal, 2358 distinct): ['117906', '240983', '117913', '279443', '117886', '130134', '117897', '117879', '168365', '133686']
ROLE_FAMILY (nominal, 67 distinct): ['290919', '118424', '19721', '117887', '292795', '118398', '308574', '118453', '118331', '118643']
ROLE_CODE (nominal, 343 distinct): ['118322', '117908', '118786', '117880', '118570', '117888', '118055', '118687', '118779', '118454']
'''

CONTEXT = "Amazon Employee Access"
TARGET = CuratedTarget(raw_name="target", new_name="Resource Access", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "Approved", "0": "Denied"})
COLS_TO_DROP = []
FEATURES = []