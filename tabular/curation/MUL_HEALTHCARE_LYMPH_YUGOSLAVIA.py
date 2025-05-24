from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: lymph
====
Examples: 148
====
URL: https://www.openml.org/search?type=data&id=10
====
Description: **Author**:   
**Source**: Unknown -   
**Please cite**:   

Citation Request:
    This lymphography domain was obtained from the University Medical Centre,
    Institute of Oncology, Ljubljana, Yugoslavia.  Thanks go to M. Zwitter and 
    M. Soklic for providing the data.  Please include this citation if you plan
    to use this database.
 
 1. Title: Lymphography Domain
 
 2. Sources: 
     (a) See Above.
     (b) Donors: Igor Kononenko, 
                 University E.Kardelj
                 Faculty for electrical engineering
                 Trzaska 25
                 61000 Ljubljana (tel.: (38)(+61) 265-161
 
                 Bojan Cestnik
                 Jozef Stefan Institute
                 Jamova 39
                 61000 Ljubljana
                 Yugoslavia (tel.: (38)(+61) 214-399 ext.287) 
    (c) Date: November 1988
 
 3. Past Usage: (sveral)
     1. Cestnik,G., Konenenko,I, & Bratko,I. (1987). Assistant-86: A
        Knowledge-Elicitation Tool for Sophisticated Users.  In I.Bratko
        & N.Lavrac (Eds.) Progress in Machine Learning, 31-45, Sigma Press.
        -- Assistant-86: 76% accuracy
     2. Clark,P. & Niblett,T. (1987). Induction in Noisy Domains.  In
        I.Bratko & N.Lavrac (Eds.) Progress in Machine Learning, 11-30,
        Sigma Press.
        -- Simple Bayes: 83% accuracy
        -- CN2 (99% threshold): 82%
     3. Michalski,R., Mozetic,I. Hong,J., & Lavrac,N. (1986).  The Multi-Purpose
        Incremental Learning System AQ15 and its Testing Applications to Three
        Medical Domains.  In Proceedings of the Fifth National Conference on
        Artificial Intelligence, 1041-1045. Philadelphia, PA: Morgan Kaufmann.
        -- Experts: 85% accuracy (estimate)
        -- AQ15: 80-82%
 
 4. Relevant Information:
      This is one of three domains provided by the Oncology Institute
      that has repeatedly appeared in the machine learning literature.
      (See also breast-cancer and primary-tumor.)
 
 5. Number of Instances: 148
 
 6. Number of Attributes: 19 including the class attribute
 
 7. Attribute information:
     --- NOTE: All attribute values in the database have been entered as
               numeric values corresponding to their index in the list
               of attribute values for that attribute domain as given below.
     1. class: normal find, metastases, malign lymph, fibrosis
     2. lymphatics: normal, arched, deformed, displaced
     3. block of affere: no, yes
     4. bl. of lymph. c: no, yes
     5. bl. of lymph. s: no, yes
     6. by pass: no, yes
     7. extravasates: no, yes
     8. regeneration of: no, yes
     9. early uptake in: no, yes
    10. lym.nodes dimin: 0-3
    11. lym.nodes enlar: 1-4
    12. changes in lym.: bean, oval, round
    13. defect in node: no, lacunar, lac. marginal, lac. central
    14. changes in node: no, lacunar, lac. margin, lac. central
    15. changes in stru: no, grainy, drop-like, coarse, diluted, reticular, 
                         stripped, faint, 
    16. special forms: no, chalices, vesicles
    17. dislocation of: no, yes
    18. exclusion of no: no, yes
    19. no. of nodes in: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70
 
 8. Missing Attribute Values: None
 
 9. Class Distribution: 
     Class:        Number of Instances:
     normal find:  2
     metastases:   81
     malign lymph: 61
     fibrosis:     4
 
 




 Relabeled values in attribute 'lymphatics'
    From: '1'                     To: normal              
    From: '2'                     To: arched              
    From: '3'                     To: deformed            
    From: '4'                     To: displaced           


 Relabeled values in attribute 'block_of_affere'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'bl_of_lymph_c'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'bl_of_lymph_s'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'by_pass'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'extravasates'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'regeneration_of'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'early_uptake_in'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'changes_in_lym'
    From: '1'                     To: bean                
    From: '2'                     To: oval                
    From: '3'                     To: round               


 Relabeled values in attribute 'defect_in_node'
    From: '1'                     To: no                  
    From: '2'                     To: lacunar             
    From: '3'                     To: lac_margin          
    From: '4'                     To: lac_central         


 Relabeled values in attribute 'changes_in_node'
    From: '1'                     To: no                  
    From: '2'                     To: lacunar             
    From: '3'                     To: lac_margin          
    From: '4'                     To: lac_central         


 Relabeled values in attribute 'changes_in_stru'
    From: '1'                     To: no                  
    From: '2'                     To: grainy              
    From: '3'                     To: drop_like           
    From: '4'                     To: coarse              
    From: '5'                     To: diluted             
    From: '6'                     To: reticular           
    From: '7'                     To: stripped            
    From: '8'                     To: faint               


 Relabeled values in attribute 'special_forms'
    From: '1'                     To: no                  
    From: '2'                     To: chalices            
    From: '3'                     To: vesicles            


 Relabeled values in attribute 'dislocation_of'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'exclusion_of_no'
    From: '1'                     To: no                  
    From: '2'                     To: yes                 


 Relabeled values in attribute 'class'
    From: '1'                     To: normal              
    From: '2'                     To: metastases          
    From: '3'                     To: malign_lymph        
    From: '4'                     To: fibrosis
====
Target Variable: class (nominal, 4 distinct): ['metastases', 'malign_lymph', 'fibrosis', 'normal']
====
Features:

lymphatics (nominal, 4 distinct): ['arched', 'deformed', 'displaced', 'normal']
block_of_affere (nominal, 2 distinct): ['yes', 'no']
bl_of_lymph_c (nominal, 2 distinct): ['no', 'yes']
bl_of_lymph_s (nominal, 2 distinct): ['no', 'yes']
by_pass (nominal, 2 distinct): ['no', 'yes']
extravasates (nominal, 2 distinct): ['yes', 'no']
regeneration_of (nominal, 2 distinct): ['no', 'yes']
early_uptake_in (nominal, 2 distinct): ['yes', 'no']
lym_nodes_dimin (numeric, 3 distinct): ['1', '3', '2']
lym_nodes_enlar (numeric, 4 distinct): ['2', '3', '4', '1']
changes_in_lym (nominal, 3 distinct): ['oval', 'round', 'bean']
defect_in_node (nominal, 4 distinct): ['lac_central', 'lacunar', 'lac_margin', 'no']
changes_in_node (nominal, 4 distinct): ['lac_margin', 'lacunar', 'lac_central', 'no']
changes_in_stru (nominal, 8 distinct): ['faint', 'coarse', 'diluted', 'drop_like', 'grainy', 'stripped', 'no', 'reticular']
special_forms (nominal, 3 distinct): ['vesicles', 'chalices', 'no']
dislocation_of (nominal, 2 distinct): ['yes', 'no']
exclusion_of_no (nominal, 2 distinct): ['yes', 'no']
no_of_nodes_in (numeric, 8 distinct): ['1', '2', '3', '4', '5', '7', '6', '8']
'''

CONTEXT = "Lymphography Study in Yugoslavia"
TARGET = CuratedTarget(raw_name="class", new_name="Diagnosis", task_type=SupervisedTask.MULTICLASS,
                       label_mapping={"fibrosis": "Other - Fibrosis or Normal",
                                      "normal": "Other - Fibrosis or Normal"})
COLS_TO_DROP = []
FEATURES = []