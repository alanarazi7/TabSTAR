from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: wine-quality-white
====
Examples: 4898
====
URL: https://www.openml.org/search?type=data&id=40498
====
Description: Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples.
  The inputs include objective tests (e.g. PH values) and the output is based on sensory data
  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
  these datasets under a regression approach. The support vector machine model achieved the
  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
  analysis procedure).
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese &quot;Vinho Verde&quot; wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than
   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
   or poor wines. Also, we are not sure if all input variables are relevant. So
   it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
   feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None
====
Target Variable: Class (nominal, 7 distinct): ['4', '3', '5', '6', '2', '1', '7']
====
Features:

V1 (numeric, 68 distinct): ['6.8', '6.6', '6.4', '6.9', '6.7', '7.0', '6.5', '7.2', '7.1', '7.4']
V2 (numeric, 125 distinct): ['0.28', '0.24', '0.26', '0.25', '0.22', '0.27', '0.23', '0.2', '0.3', '0.21']
V3 (numeric, 87 distinct): ['0.3', '0.28', '0.32', '0.34', '0.29', '0.26', '0.27', '0.49', '0.31', '0.33']
V4 (numeric, 310 distinct): ['1.2', '1.4', '1.6', '1.3', '1.1', '1.5', '1.7', '1.8', '1.0', '2.0']
V5 (numeric, 160 distinct): ['0.044', '0.036', '0.042', '0.04', '0.046', '0.048', '0.047', '0.045', '0.05', '0.034']
V6 (numeric, 132 distinct): ['29.0', '31.0', '26.0', '35.0', '34.0', '36.0', '24.0', '28.0', '33.0', '25.0']
V7 (numeric, 251 distinct): ['111.0', '113.0', '117.0', '118.0', '128.0', '114.0', '150.0', '122.0', '124.0', '140.0']
V8 (numeric, 890 distinct): ['0.992', '0.9928', '0.9932', '0.993', '0.9934', '0.9938', '0.9927', '0.9944', '0.9948', '0.9954']
V9 (numeric, 103 distinct): ['3.14', '3.16', '3.22', '3.19', '3.18', '3.2', '3.15', '3.08', '3.1', '3.12']
V10 (numeric, 79 distinct): ['0.5', '0.46', '0.44', '0.38', '0.42', '0.48', '0.45', '0.47', '0.4', '0.54']
V11 (numeric, 103 distinct): ['9.4', '9.5', '9.2', '9.0', '10.0', '10.5', '11.0', '10.4', '9.1', '9.8']
'''

CONTEXT = "Wine Quality Estimation for Red and White Wine"
TARGET = CuratedTarget(raw_name="Class", new_name="Wine Quality", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="V1", new_name="Fixed Acidity"),
            CuratedFeature(raw_name="V2", new_name="Volatile Acidity"),
            CuratedFeature(raw_name="V3", new_name="Citric Acid"),
            CuratedFeature(raw_name="V4", new_name="Residual Sugar"),
            CuratedFeature(raw_name="V5", new_name="Chlorides"),
            CuratedFeature(raw_name="V6", new_name="Free Sulfur Dioxide"),
            CuratedFeature(raw_name="V7", new_name="Total Sulfur Dioxide"),
            CuratedFeature(raw_name="V8", new_name="Density"),
            CuratedFeature(raw_name="V9", new_name="pH"),
            CuratedFeature(raw_name="V10", new_name="Sulphates"),
            CuratedFeature(raw_name="V11", new_name="Alcohol")]
