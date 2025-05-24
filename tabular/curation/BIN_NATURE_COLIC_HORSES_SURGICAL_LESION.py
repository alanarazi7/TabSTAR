from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: colic
====
Examples: 368
====
URL: https://www.openml.org/search?type=data&id=25
====
Description: **Author**: Mary McLeish & Matt Cecile, University of Guelph  
Donor: Will Taylor (taylor@pluto.arc.nasa.gov)   
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Horse+Colic) - 8/6/89   

**Horse Colic database**  
Database of surgeries on horses. Possible class attributes: 24 (whether lesion is surgical), others include: 23, 25, 26, and 27

Notes:
* Hospital_Number is an identifier and should be ignored when modelling

Attribute Information:
> 
   1:  surgery?
           1 = Yes, it had surgery
           2 = It was treated without surgery  
   2:  Age 
           1 = Adult horse
           2 = Young (< 6 months)  
   3:  Hospital Number 
           - numeric id
           - the case number assigned to the horse
             (may not be unique if the horse is treated > 1 time)  
   4:  rectal temperature
           - linear
           - in degrees celsius.
           - An elevated temp may occur due to infection.
           - temperature may be reduced when the animal is in late shock
           - normal temp is 37.8
           - this parameter will usually change as the problem progresses
                eg. may start out normal, then become elevated because of
                    the lesion, passing back through the normal range as the
                    horse goes into shock  
   5:  pulse 
           - linear
           - the heart rate in beats per minute
           - is a reflection of the heart condition: 30 -40 is normal for adults
           - rare to have a lower than normal rate although athletic horses
             may have a rate of 20-25
           - animals with painful lesions or suffering from circulatory shock
             may have an elevated heart rate  
   6:  respiratory rate
           - linear
           - normal rate is 8 to 10
           - usefulness is doubtful due to the great fluctuations  
   7:  temperature of extremities
           - a subjective indication of peripheral circulation
           - possible values:
                1 = Normal
                2 = Warm
                3 = Cool
                4 = Cold
           - cool to cold extremities indicate possible shock
           - hot extremities should correlate with an elevated rectal temp.  
   8:  peripheral pulse
           - subjective
           - possible values are:
                1 = normal
                2 = increased
                3 = reduced
                4 = absent
           - normal or increased p.p. are indicative of adequate circulation
             while reduced or absent indicate poor perfusion  
   9:  mucous membranes
           - a subjective measurement of colour
           - possible values are:
                1 = normal pink
                2 = bright pink
                3 = pale pink
                4 = pale cyanotic
                5 = bright red / injected
                6 = dark cyanotic
           - 1 and 2 probably indicate a normal or slightly increased
             circulation
           - 3 may occur in early shock
           - 4 and 6 are indicative of serious circulatory compromise
           - 5 is more indicative of a septicemia  
  10: capillary refill time
           - a clinical judgement. The longer the refill, the poorer the
             circulation
           - possible values
                1 = < 3 seconds
                2 = >= 3 seconds  
  11: pain - a subjective judgement of the horse's pain level
           - possible values:
                1 = alert, no pain
                2 = depressed
                3 = intermittent mild pain
                4 = intermittent severe pain
                5 = continuous severe pain
           - should NOT be treated as a ordered or discrete variable!
           - In general, the more painful, the more likely it is to require
             surgery
           - prior treatment of pain may mask the pain level to some extent  
  12: peristalsis                              
           - an indication of the activity in the horse's gut. As the gut
             becomes more distended or the horse becomes more toxic, the
             activity decreases
           - possible values:
                1 = hypermotile
                2 = normal
                3 = hypomotile
                4 = absent  
  13: abdominal distension
           - An IMPORTANT parameter.
           - possible values
                1 = none
                2 = slight
                3 = moderate
                4 = severe
           - an animal with abdominal distension is likely to be painful and
             have reduced gut motility.
           - a horse with severe abdominal distension is likely to require
             surgery just tio relieve the pressure  
  14: nasogastric tube
           - this refers to any gas coming out of the tube
           - possible values:
                1 = none
                2 = slight
                3 = significant
           - a large gas cap in the stomach is likely to give the horse
             discomfort  
  15: nasogastric reflux
           - possible values
                1 = none
                2 = > 1 liter
                3 = < 1 liter
           - the greater amount of reflux, the more likelihood that there is
             some serious obstruction to the fluid passage from the rest of
             the intestine  
  16: nasogastric reflux PH
           - linear
           - scale is from 0 to 14 with 7 being neutral
           - normal values are in the 3 to 4 range  
  17: rectal examination - feces
           - possible values
                1 = normal
                2 = increased
                3 = decreased
                4 = absent
           - absent feces probably indicates an obstruction  
  18: abdomen
           - possible values
                1 = normal
                2 = other
                3 = firm feces in the large intestine
                4 = distended small intestine
                5 = distended large intestine
           - 3 is probably an obstruction caused by a mechanical impaction
             and is normally treated medically
           - 4 and 5 indicate a surgical lesion  
  19: packed cell volume
           - linear
           - the # of red cells by volume in the blood
           - normal range is 30 to 50. The level rises as the circulation
             becomes compromised or as the animal becomes dehydrated.  
  20: total protein
           - linear
           - normal values lie in the 6-7.5 (gms/dL) range
           - the higher the value the greater the dehydration  
  21: abdominocentesis appearance
           - a needle is put in the horse's abdomen and fluid is obtained from
             the abdominal cavity
           - possible values:
                1 = clear
                2 = cloudy
                3 = serosanguinous
           - normal fluid is clear while cloudy or serosanguinous indicates
             a compromised gut  
  22: abdomcentesis total protein
           - linear
           - the higher the level of protein the more likely it is to have a
             compromised gut. Values are in gms/dL  
  23: outcome
           - what eventually happened to the horse?
           - possible values:
                1 = lived
                2 = died
                3 = was euthanized  
  24: surgical lesion?
           - retrospectively, was the problem (lesion) surgical?
           - all cases are either operated upon or autopsied so that
             this value and the lesion type are always known
           - possible values:
                1 = Yes
                2 = No  
  25, 26, 27: type of lesion
           - first number is site of lesion
                1 = gastric
                2 = sm intestine
                3 = lg colon
                4 = lg colon and cecum
                5 = cecum
                6 = transverse colon
                7 = retum/descending colon
                8 = uterus
                9 = bladder
                11 = all intestinal sites
                00 = none
           - second number is type
                1 = simple
                2 = strangulation
                3 = inflammation
                4 = other
           - third number is subtype
                1 = mechanical
                2 = paralytic
                0 = n/a
           - fourth number is specific code
                1 = obturation
                2 = intrinsic
                3 = extrinsic
                4 = adynamic
                5 = volvulus/torsion
                6 = intussuption
                7 = thromboembolic
                8 = hernia
                9 = lipoma/slenic incarceration
                10 = displacement
                0 = n/a
  28: cp_data
           - is pathology data present for this case?
                1 = Yes
                2 = No
           - this variable is of no significance since pathology data
             is not included or collected for these cases
====
Target Variable: surgical_lesion (nominal, 2 distinct): ['1', '2']
====
Features:

surgery (nominal, 3 distinct): ['1', '2']
Age (nominal, 2 distinct): ['1', '9']
rectal_temperature (numeric, 41 distinct): ['38.0', '38.3', '38.2', '38.5', '37.8', '37.5', '38.1', '38.4', '38.6', '37.6']
pulse (numeric, 55 distinct): ['48.0', '60.0', '40.0', '44.0', '42.0', '88.0', '52.0', '100.0', '72.0', '120.0']
respiratory_rate (numeric, 41 distinct): ['20.0', '24.0', '12.0', '30.0', '16.0', '40.0', '36.0', '28.0', '32.0', '60.0']
temperature_of_extremities (nominal, 5 distinct): ['3', '1', '2', '4']
peripheral_pulse (nominal, 5 distinct): ['1', '3', '4', '2']
mucous_membranes (nominal, 7 distinct): ['1', '3', '4', '2', '5', '6']
capillary_refill_time (nominal, 4 distinct): ['1', '2', '3']
pain (nominal, 6 distinct): ['3', '2', '5', '1', '4']
peristalsis (nominal, 5 distinct): ['3', '4', '1', '2']
abdominal_distension (nominal, 5 distinct): ['1', '3', '2', '4']
nasogastric_tube (nominal, 4 distinct): ['2', '1', '3']
nasogastric_reflux (nominal, 4 distinct): ['1', '3', '2']
nasogastric_reflux_PH (numeric, 25 distinct): ['2.0', '7.0', '5.0', '6.5', '6.0', '5.5', '3.0', '7.5', '4.0', '4.5']
rectal_examination_-_feces (nominal, 5 distinct): ['4', '1', '3', '2']
abdomen (nominal, 6 distinct): ['5', '4', '1', '2', '3']
packed_cell_volume (numeric, 55 distinct): ['37.0', '45.0', '44.0', '43.0', '35.0', '50.0', '40.0', '47.0', '36.0', '48.0']
total_protein (numeric, 85 distinct): ['7.5', '6.5', '7.0', '6.6', '6.0', '65.0', '5.9', '6.7', '7.2', '6.8']
abdominocentesis_appearance (nominal, 4 distinct): ['2', '3', '1']
abdomcentesis_total_protein (numeric, 45 distinct): ['2.0', '1.0', '3.9', '5.0', '2.6', '2.8', '3.6', '7.0', '4.3', '3.4']
outcome (nominal, 4 distinct): ['1', '2', '3']
site_of_lesion (nominal, 63 distinct): ['0', '3111', '3205', '2208', '2205', '2209', '4205', '7111', '1400', '2124']
type_of_lesion (nominal, 8 distinct): ['0', '3111', '3205', '1400', '7111', '6112', '3112', '2208']
subtype_of_lesion (nominal, 2 distinct): ['0', '2209']
pathology_cp_data (nominal, 2 distinct): ['2', '1']
'''

CONTEXT = "Colic Surgery on Horses"
TARGET = CuratedTarget(raw_name="surgical_lesion", new_name="Surgical Lesion", task_type=SupervisedTask.BINARY,
                       label_mapping={"1": "Yes", "2": "No"})
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="surgery", new_name="Treated with surgery", value_mapping={"1": "Yes", "2": "No"}),
            CuratedFeature(raw_name="Age", new_name="Age", value_mapping={"1": "Adult", "9": "Younger than 6 months"}),
            CuratedFeature(raw_name="temperature_of_extremities",
                           value_mapping={"1": "Normal", "2": "Warm", "3": "Cool", "4": "Cold"}),
            CuratedFeature(raw_name="peripheral_pulse",
                           value_mapping={"1": "Normal", "2": "Increased", "3": "Reduced", "4": "Absent"}),
            CuratedFeature(raw_name="mucous_membranes",
                           value_mapping={"1": "Normal pink", "2": "Bright pink", "3": "Pale pink", "4": "Pale cyanotic",
                                          "5": "Bright red / injected", "6": "Dark cyanotic"}),
            CuratedFeature(raw_name="pain",
                           value_mapping={"1": "Alert, no pain", "2": "Depressed", "3": "Intermittent mild pain",
                                          "4": "Intermittent severe pain", "5": "Continuous severe pain"}),
            CuratedFeature(raw_name="peristalsis",
                           value_mapping={"1": "Hypermotile", "2": "Normal", "3": "Hypomotile", "4": "Absent"}),
            CuratedFeature(raw_name="abdominal_distension",
                           value_mapping={"1": "None", "2": "Slight", "3": "Moderate", "4": "Severe"}),
            CuratedFeature(raw_name="nasogastric_tube",
                           value_mapping={"1": "None", "2": "Slight", "3": "Significant"}),
            CuratedFeature(raw_name="nasogastric_reflux",
                           value_mapping={"1": "None", "2": "> 1 liter", "3": "< 1 liter"}),
            CuratedFeature(raw_name="rectal_examination_-_feces",
                           value_mapping={"1": "Normal", "2": "Increased", "3": "Decreased", "4": "Absent"}),
            CuratedFeature(raw_name="abdomen",
                           value_mapping={"1": "Normal", "2": "Other", "3": "Firm feces in the large intestine",
                                          "4": "Distended small intestine", "5": "Distended large intestine"}),
            CuratedFeature(raw_name="abdominocentesis_appearance",
                           value_mapping={"1": "Clear", "2": "Cloudy", "3": "Serosanguinous"}),
            CuratedFeature(raw_name="outcome",
                           value_mapping={"1": "Lived", "2": "Died", "3": "Euthanized"}),
            CuratedFeature(raw_name="pathology_cp_data", value_mapping={"1": "Yes", "2": "No"}),
            ]
