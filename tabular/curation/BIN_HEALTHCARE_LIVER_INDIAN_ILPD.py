from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: ilpd
====
Examples: 583
====
URL: https://www.openml.org/search?type=data&id=1480
====
Description: **Author**: Bendi Venkata Ramana, M. Surendra Prasad Babu, N. B. Venkateswarlu  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)) - 2012  
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  

**Indian Liver Patient Dataset**  
This data set contains 416 liver patient records and 167 non liver patient records.The data set was collected from north east of Andhra Pradesh, India. The class label divides the patients into 2 groups (liver patient or not). This data set contains 441 male patient records and 142 female patient records. 

### Attribute Information  
V1. Age of the patient. Any patient whose age exceeded 89 is listed as being of age "90".  
V2. Gender of the patient  
V3. Total Bilirubin  
V4. Direct Bilirubin  
V5. Alkphos Alkaline Phosphatase  
V6. Sgpt Alanine Aminotransferase  
V7. Sgot Aspartate Aminotransferase   
V8. Total Proteins  
V9. Albumin  
V10. A/G Ratio Albumin and Globulin Ratio  

A feature indicating a train-test split has been removed.  

### Relevant Papers  
1. Bendi Venkata Ramana, Prof. M. S. Prasad Babu and Prof. N. B. Venkateswarlu, A Critical Comparative Study of Liver Patients from USA and INDIA: An Exploratory Analysis, International Journal of Computer Science Issues, ISSN:1694-0784, May 2012. 
2. Bendi Venkata Ramana, Prof. M. S. Prasad Babu and Prof. N. B. Venkateswarlu, A Critical Study of Selected Classification Algorithms for Liver Disease Diagnosis, International Journal of Database Management Systems (IJDMS), Vol.3, No.2, ISSN : 0975-5705, PP 101-114, May 2011.
====
Target Variable: Class (nominal, 2 distinct): ['1', '2']
====
Features:

V1 (numeric, 72 distinct): ['60', '45', '50', '42', '38', '32', '48', '55', '65', '40']
V2 (nominal, 2 distinct): ['Male', 'Female']
V3 (numeric, 113 distinct): ['0.8', '0.7', '0.9', '0.6', '1.0', '1.1', '1.8', '1.4', '1.3', '1.7']
V4 (numeric, 80 distinct): ['0.2', '0.1', '0.3', '0.8', '0.4', '0.5', '0.6', '1.0', '1.3', '0.7']
V5 (numeric, 263 distinct): ['198.0', '215.0', '298.0', '195.0', '190.0', '180.0', '145.0', '158.0', '182.0', '282.0']
V6 (numeric, 152 distinct): ['25.0', '20.0', '22.0', '28.0', '21.0', '18.0', '30.0', '48.0', '15.0', '24.0']
V7 (numeric, 177 distinct): ['23.0', '30.0', '20.0', '21.0', '22.0', '28.0', '25.0', '34.0', '24.0', '32.0']
V8 (numeric, 58 distinct): ['7.0', '6.0', '6.8', '6.9', '6.2', '7.1', '7.2', '8.0', '7.3', '5.6']
V9 (numeric, 40 distinct): ['3.0', '4.0', '2.9', '3.1', '3.2', '3.9', '2.7', '2.5', '3.5', '2.6']
V10 (numeric, 70 distinct): ['1.0', '0.8', '0.9', '0.7', '1.1', '1.2', '0.6', '0.5', '1.3', '1.4']
'''

CONTEXT = "Indian Liver Patient Dataset"
TARGET = CuratedTarget(raw_name="Class", new_name="Liver Patient", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "Liver Patient", "2": "Not Liver Patient"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Age"),
            CuratedFeature(raw_name="V2", new_name="Gender"),
            CuratedFeature(raw_name="V3", new_name="Total Bilirubin"),
            CuratedFeature(raw_name="V4", new_name="Direct Bilirubin"),
            CuratedFeature(raw_name="V5", new_name="Alkphos Alkaline Phosphatase"),
            CuratedFeature(raw_name="V6", new_name="Sgpt Alanine Aminotransferase"),
            CuratedFeature(raw_name="V7", new_name="Sgot Aspartate Aminotransferase"),
            CuratedFeature(raw_name="V8", new_name="Total Proteins"),
            CuratedFeature(raw_name="V9", new_name="Albumin"),
            CuratedFeature(raw_name="V10", new_name="A/G Ratio Albumin and Globulin Ratio")]