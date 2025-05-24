from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: irish
====
Examples: 500
====
URL: https://www.openml.org/search?type=data&id=451
====
Description: **Author**: Vincent Greaney, Thomas Kelleghan (St. Patrick's College, Dublin)   
**Source**: [StatLib](http://lib.stat.cmu.edu/datasets/irish.ed) - 1984  
**Please cite**: [StatLib](http://lib.stat.cmu.edu/datasets/)

**Irish Educational Transitions Data**  
Data on educational transitions for a sample of 500 Irish schoolchildren aged 11 in 1967. The data were collected by Greaney and Kelleghan (1984), and reanalyzed by Raftery and Hout (1985, 1993). 

### Attribute information  

* Sex: 1=male; 2=female.
* DVRT (Drumcondra Verbal Reasoning Test Score).
* Educational level attained
* Leaving Certificate. 1 if Leaving Certificate not taken; 2 if taken.
* Prestige score for father's occupation (calculated by Raftery and Hout, 1985).
* Type of school: 1=secondary; 2=vocational; 9=primary terminal leaver.

### Relevant papers  

Greaney, V. and Kelleghan, T. (1984). Equality of Opportunity in Irish
Schools. Dublin: Educational Company.

Kass, R.E. and Raftery, A.E. (1993). Bayes factors and model uncertainty.
Technical Report no. 254, Department of Statistics, University of Washington.
Revised version to appear in Journal of the American Statistical
Association.

Raftery, A.E. (1988). Approximate Bayes factors for generalized linear models.
Technical Report no. 121, Department of Statistics, University of Washington.

Raftery, A.E. and Hout, M. (1985). Does Irish education approach the
meritocratic ideal? A logistic analysis.
Economic and Social Review, 16, 115-140.

Raftery, A.E. and Hout, M. (1993). Maximally maintained inequality:
Expansion, reform and opportunity in Irish schools.
Sociology of Education, 66, 41-62.


### Ownership Statement  
This data belongs to Vincent Greaney and Thomas Kelleghan, Educational Research Centre, St. Patrick's College, Drumcondra, Dublin 9, Ireland, who retain the copyright.

In the form given here, it may be used solely as an example for research on the development of statistical methods. For any other use of the data, permission must be obtained from the owners.
====
Target Variable: Leaving_Certificate (nominal, 2 distinct): ['not_taken', 'taken']
====
Features:

Sex (nominal, 2 distinct): ['female', 'male']
DVRT (numeric, 68 distinct): ['70', '104', '90', '103', '100', '109', '114', '99', '91', '94']
Educational_level (nominal, 11 distinct): ['Senior_cycle_terminal_leaver-secondary_school', 'Junior_cycle_terminal_leaver-vocational_school', 'Junior_cycle_terminal_leaver-secondary_school', '3rd_level_complete', 'Junior_cycle_incomplete-vocational_school', 'Primary_terminal_leaver', 'Junior_cycle_incomplete-secondary_school', 'Senior_cycle_incomplete-vocational_school', 'Senior_cycle_incomplete-secondary_school', '3rd_level_incomplete']
Prestige_score (numeric, 29 distinct): ['18.0', '37.0', '43.0', '28.0', '58.0', '40.0', '57.0', '35.0', '61.0', '31.0']
Type_school (nominal, 3 distinct): ['secondary', 'vocational', 'primary_terminal_leaver']
'''

CONTEXT = "Educational transitions in Irish schoolchildren aged 11 in 1967"
TARGET = CuratedTarget(raw_name="Leaving_Certificate", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []