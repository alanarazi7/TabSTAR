from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: adult
====
Examples: 48842
====
URL: https://www.openml.org/search?type=data&id=1590
====
Description: **Author**: Ronny Kohavi and Barry Becker  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Adult) - 1996  
**Please cite**: Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996  

Prediction task is to determine whether a person makes over 50K a year. Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

This is the original version from the UCI repository, with training and test sets merged.

### Variable description

Variables are all self-explanatory except __fnlwgt__. This is a proxy for the demographic background of the people: "People with similar demographic characteristics should have similar weights". This similarity-statement is not transferable across the 51 different states.

Description from the donor of the database: 

The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US.  These are prepared monthly for us by Population Division here at the Census Bureau. We use 3 sets of controls. These are:
1.  A single cell estimate of the population 16+ for each state.
2.  Controls for Hispanic Origin by age and sex.
3.  Controls by Race, age and sex.

We use all three sets of controls in our weighting program and "rake" through them 6 times so that by the end we come back to all the controls we used. The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population. People with similar demographic characteristics should have similar weights. There is one important caveat to remember about this statement. That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.


### Relevant papers  

Ronny Kohavi and Barry Becker. Data Mining and Visualization, Silicon Graphics.  
e-mail: ronnyk '@' live.com for questions.
====
Target Variable: class (nominal, 2 distinct): ['<=50K', '>50K']
====
Features:

age (numeric, 74 distinct): ['36', '35', '33', '23', '31', '34', '37', '28', '30', '38']
workclass (nominal, 9 distinct): ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
fnlwgt (numeric, 28523 distinct): ['203488.0', '190290.0', '120277.0', '125892.0', '126569.0', '99185.0', '126675.0', '113364.0', '186934.0', '111567.0']
education (nominal, 16 distinct): ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school']
education-num (numeric, 16 distinct): ['9', '10', '13', '14', '11', '7', '12', '6', '4', '15']
marital-status (nominal, 7 distinct): ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation (nominal, 15 distinct): ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing']
relationship (nominal, 6 distinct): ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race (nominal, 5 distinct): ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
sex (nominal, 2 distinct): ['Male', 'Female']
capital-gain (numeric, 123 distinct): ['0.0', '15024.0', '7688.0', '7298.0', '99999.0', '3103.0', '5178.0', '5013.0', '4386.0', '8614.0']
capital-loss (numeric, 99 distinct): ['0.0', '1902.0', '1977.0', '1887.0', '2415.0', '1485.0', '1848.0', '1590.0', '1602.0', '1876.0']
hours-per-week (numeric, 96 distinct): ['40', '50', '45', '60', '35', '20', '30', '55', '25', '48']
native-country (nominal, 42 distinct): ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England']
'''

CONTEXT = "USA Adult Citizens Income Prediction"
TARGET = CuratedTarget(raw_name="class", new_name="Income", task_type=SupervisedTask.BINARY,
                       label_mapping={'<=50K': 'Lower than 50K', '>50K': 'Higher than 50K'})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="fnlwgt", new_name="Proxy for Demographic Background")]
