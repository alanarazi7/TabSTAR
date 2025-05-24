from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: The-Estonia-Disaster-Passenger-List
====
Examples: 989
====
URL: https://www.openml.org/search?type=data&id=43389
====
Description: Introduction
On September 27 1994 the ferry Estonia set sail on a night voyage across the Baltic Sea from the port of Tallin in Estonia to Stockholm. She departed at 19.00 carrying 989 passengers and crew, as well as vehicles, and was due to dock at 09.30 the following morning, Tragically, the Estonia never arrived.
The weather was typically stormy for the time of year but, like all the other scheduled ferries on that day, the Estonia set off as usual. At roughly 01:00 a worrying sound of screeching metal was heard, but an immediate inspection of the bow visor showed nothing untoward. The ship suddenly listed 15 minutes later and soon alarms were sounding, including the lifeboat alarm. Shortly afterwards the Estonia rolled drastically to starboard. Those who had reached the decks had a chance of survival but those who had not were doomed as the angled corridors had become death traps. A Mayday signal was sent but power failure meant the ships position was given imprecisely. The Estonia disappeared from the responding ships radar screens at about 01:50.
The Marietta arrived at the scene at 02:12 and the first helicopter at 03:05. Of the 138 people rescued alive, one died later in hospital.
Of the 310 people who had reached the decks, almost a third died of hypothermia. The final death toll was shockingly high  more than 850 people.
An official inquiry found that failure of the locks on the bow visor, which broke away under the punishing waves, caused water to flood the car deck and quickly capsize the ship. The report also noted a lack of action, delay in sounding the alarm, lack of guidance from the bridge and a failure to light distress flares.
The sinking of the Estonia was Europes worst postwar maritime disaster.

Read more: https://en.wikipedia.org/wiki/MS_Estonia

Facts
When was the Sinking of the Estonia: September 28, 1994
Where was the Sinking of the Estonia: Near the Turku Archipelago, in the Baltic Sea
What was the Sinking of the Estonia death toll: 852 passengers and crew

Interesting things to investigate about the data:

Who's more likely to survive the sinking based on data?
Is age an indicator for survival?
Is gender an indicator for survival?
Did the crew aboard have a higher chance of survival than passengers?
Since the death toll is well above 80, can you make a classifier that beats the baseline (all passengers died)?

Video
Watch the Zero Hour documentary about the disaster: https://www.youtube.com/watch?v=eFDGL_ehpkI
====
Target Variable: Survived (numeric, 2 distinct): ['0', '1']
====
Features:

Country (string, 16 distinct): ['Sweden', 'Estonia', 'Latvia', 'Finland', 'Russia', 'Norway', 'Germany', 'Denmark', 'Lithuania', 'Great Britain']
Firstname (string, 849 distinct): ['ANDRES', 'RAIVO', 'TIINA', 'PEETER', 'KATRIN', 'PAUL', 'ULLE', 'ANDRUS', 'SIRJE', 'LIA']
Lastname (string, 774 distinct): ['ANDERSSON', 'NILSSON', 'ERIKSSON', 'JOHANSSON', 'KARLSSON', 'GUSTAFSSON', 'PERSSON', 'SVENSSON', 'JONSSON', 'PETTERSSON']
Sex (string, 2 distinct): ['M', 'F']
Age (numeric, 78 distinct): ['21', '45', '30', '67', '50', '49', '24', '41', '37', '43']
Category (string, 2 distinct): ['P', 'C']
'''

CONTEXT = "Estonian Ferry Disaster Passenger List"
TARGET = CuratedTarget(raw_name="Survived", task_type=SupervisedTask.BINARY,
                       label_mapping={'0': 'Died', '1': 'Survived'})
COLS_TO_DROP = []
FEATURES = [
            CuratedFeature(raw_name="Firstname", new_name="First Name"),
            CuratedFeature(raw_name="Lastname", new_name="Last Name"),
            CuratedFeature(raw_name="Sex", value_mapping={"M": "Male", "F": "Female"}),
            CuratedFeature(raw_name="Category", value_mapping={"P": "Passenger", "C": "Crew"})
            ]
