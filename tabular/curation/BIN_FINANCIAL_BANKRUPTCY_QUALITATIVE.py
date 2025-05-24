from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''Dataset Name: qualitative-bankruptcy
====
Examples: 250
====
URL: https://www.openml.org/search?type=data&id=1495
====
Description: **Author**:  A. Martin, J. Uthayakumar, M. Nadarajan, V. Prasanna Venkatesan   
**Source**: UCI   
**Please cite**:    

* Abstract: 

Predict the Bankruptcy from Qualitative parameters from experts.

* Source:

Source Information
-- Creator : Mr.A.Martin(jayamartin '@' yahoo.com)
Mr.J.Uthayakumar (uthayakumar17691 '@' gmail.com)
Mr.M.Nadarajan(nadaraj.muthuvel '@' gmail.com)
-- Guided By : Dr.V.Prasanna Venkatesan
-- Institution : Sri Manakula Vinayagar Engineering College and Pondicherry University
-- Country : India
-- Date : February 2014


* Data Set Information:

The parameters which we used for collecting the dataset is referred from the paper 'The discovery of expert' decision rules from qualitative bankruptcy data using genetic algorithms' by Myoung-Jong Kim*, Ingoo Han.


* Attribute Information: 
(P=Positive,A-Average,N-negative,B-Bankruptcy,NB-Non-Bankruptcy) 

1. Industrial Risk: {P,A,N} 
2. Management Risk: {P,A,N} 
3. Financial Flexibility: {P,A,N} 
4. Credibility: {P,A,N} 
5. Competitiveness: {P,A,N} 
6. Operating Risk: {P,A,N} 
7. Class: {B,NB}


* Relevant Papers:

The parameters which we used for collecting the dataset is referred from the paper 'The discovery of expertsâ€™ decision rules from qualitative bankruptcy data using genetic algorithms' by Myoung-Jong Kim*, Ingoo Han.
====
Target Variable: Class (nominal, 2 distinct): ['2', '1']
====
Features:

V1 (nominal, 3 distinct): ['2', '1', '3']
V2 (nominal, 3 distinct): ['2', '1', '3']
V3 (nominal, 3 distinct): ['2', '1', '3']
V4 (nominal, 3 distinct): ['2', '3', '1']
V5 (nominal, 3 distinct): ['2', '3', '1']
V6 (nominal, 3 distinct): ['2', '3', '1']
'''

CONTEXT = "Bankruptcy from Qualitative parameters from experts"
TARGET = CuratedTarget(raw_name="Class", new_name="Bankruptcy", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Industrial Risk"),
            CuratedFeature(raw_name="V2", new_name="Management Risk"),
            CuratedFeature(raw_name="V3", new_name="Financial Flexibility"),
            CuratedFeature(raw_name="V4", new_name="Credibility"),
            CuratedFeature(raw_name="V5", new_name="Competitiveness"),
            CuratedFeature(raw_name="V6", new_name="Operating Risk")]
