from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: mozilla4
====
Examples: 15545
====
URL: https://www.openml.org/search?type=data&id=1046
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This is a PROMISE Software Engineering Repository data set made publicly
available in order to encourage repeatable, verifiable, refutable, and/or
improvable predictive models of software engineering.

If you publish material based on PROMISE data sets then, please
follow the acknowledgment guidelines posted on the PROMISE repository
web page http://promisedata.org/repository .
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(c) 2007  A. Gunes Koru
Contact: gkoru AT umbc DOT edu Phone: +1 (410) 455 8843
This data set is distributed under the
Creative Commons Attribution-Share Alike 3.0 License
http://creativecommons.org/licenses/by-sa/3.0/

You are free:

* to Share -- copy, distribute and transmit the work
* to Remix -- to adapt the work

Under the following conditions:

Attribution. You must attribute the work in the manner specified by
the author or licensor (but not in any way that suggests that they endorse
you or your use of the work).

Share Alike. If you alter, transform, or build upon this work, you
may distribute the resulting work only under the same, similar or a
compatible license.

* For any reuse or distribution, you must make clear to others the
license terms of this work.
* Any of the above conditions can be waived if you get permission from
the copyright holder.
* Apart from the remix rights granted under this license, nothing in
this license impairs or restricts the author's moral rights.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


1. Title: Recurrent event (defect fix) and size data for Mozilla Classes
This one includes a binary attribute (event) to show defect fix.
The data is at the "observation" level. Each modification made to
a C++ class was entered as an observation. A newly added class
created an observation. The observation period was between
May 29, 2002 and Feb 22, 2006.

2. Sources
(a) Creator: A. Gunes Koru
(b) Date: February 23, 2007
(c) Contact: gkoru AT umbc DOT edu Phone: +1 (410) 455 8843

3. Donor: A. Gunes Koru

4. Past Usage: This data set was used for:

A. Gunes Koru, Dongsong Zhang, and Hongfang Liu, "Modeling the
Effect of Size on Defect Proneness for Open-Source Software",
Predictive Models in Software Engineering Workshop, PROMISE 2007,
May 20th 2007, Minneapolis, Minnesota, US.

Abstract:
Quality is becoming increasingly important with the continuous
adoption of open-source software.  Previous research has found that
there is generally a positive relationship between module size and
defect proneness. Therefore, in open-source software development, it
is important to monitor module size and understand its impact on
defect proneness. However, traditional approaches to quality
modeling, which measure specific system snapshots and obtain future
defect counts, are not well suited because open-source modules
usually evolve and their size changes over time. In this study, we
used Cox proportional hazards modeling with recurrent events to
study the effect of class size on defect-proneness in the Mozilla
product. We found that the effect of size was significant, and we
quantified this effect on defect proneness.

The full paper can be downloaded from A. Gunes Koru's Website
http://umbc.edu/~gkoru
by following the Publications link or from the Web site of PROMISE 2007.

5. Features:

This data set is used to create a conditional Cox Proportional
Hazards Model

id: A numeric identification assigned to each separate C++ class
(Note that the id's do not increment from the first to the last
data row)

start: A time infinitesimally greater than the time of the modification
that created this observation (practically, modification time). When a
class is introduced to a system, a new observation is entered with start=0

end: Either the time of the next modification, or the end of the
observation period, or the time of deletion, whichever comes first.

event: event is set to 1 if a defect fix takes place
at the time represented by 'end', or 0 otherwise.  A class deletion
is handled easily by entering a final observation whose event is set
to 1 if the class is deleted for corrective maintenance, or 0 otherwise.

size: It is a time-dependent covariate and its column carries the
number of source Lines of Code of the C++ classes
at time 'start'. Blank and comment lines are not counted.

state: Initially set to 0, and it becomes 1 after the class
experiences an event, and remains at 1 thereafter.
====
Target Variable: state (nominal, 2 distinct): ['1', '0']
====
Features:

id (numeric, 4089 distinct): ['500.0', '3187.0', '2648.0', '3506.0', '498.0', '3534.0', '3415.0', '1581.0', '1804.0', '1873.0']
start (numeric, 8525 distinct): ['0.0', '1338416.0', '1717382.0', '907686.0', '1717381.0', '1286652.0', '1338415.0', '6230.0', '1338414.0', '1717380.0']
end (numeric, 10599 distinct): ['163760.0', '1338416.0', '1717382.0', '475147.0', '36355.0', '1965726.0', '907686.0', '1717381.0', '1371993.0', '1286652.0']
event (numeric, 2 distinct): ['1', '0']
size (numeric, 2000 distinct): ['10.0', '6.0', '7.0', '15.0', '17.0', '5.0', '13.0', '22.0', '12.0', '11.0']
'''

CONTEXT = "Mozilla Classes"
TARGET = CuratedTarget(raw_name='state', new_name="Had event", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = []