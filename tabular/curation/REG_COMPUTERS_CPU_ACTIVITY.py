from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: cpu_activity
====
Examples: 8192
====
URL: https://www.openml.org/search?type=data&id=44978
====
Description: **Data Description**


The Computer Activity databases are a collection of computer systems activity measures. The data was collected from a Sun Sparcstation 20/712 with 128 Mbytes of memory running in a multi-user university department.
Users would typically be doing a large variety of tasks ranging from accessing the internet, editing files or running very cpu-bound programs. The data was collected continuously on two separate occasions. On both occassions, system activity was gathered every 5 seconds. The final dataset is taken from both occasions with equal numbers of observations coming from each collection epoch.

Each instance represents one observed system performance. The goal is to predict portion of time that cpus run in user mode.

**Attribute Description**

1. *lread* - Reads (transfers per second ) between system memory and user memory.
2. *lwrite* - writes (transfers per second) between system memory and user memory.
3. *scall* - Number of system calls of all types per second.
4. *sread* - Number of system read calls per second.
5. *swrite* - Number of system write calls per second .
6. *fork* - Number of system fork calls per second.
7. *exec* - Number of system exec calls per second.
8. *rchar* - Number of characters transferred per second by system read calls.
9. *wchar* - Number of characters transfreed per second by system write calls.
10. *pgout* - Number of page out requests per second.
11. *ppgout* - Number of pages, paged out per second.
12. *pgfree* - Number of pages per second placed on the free list.
13. *pgscan* - Number of pages checked if they can be freed per second.
14. *atch* - Number of page attaches (satisfying a page fault by reclaiming a page in memory) per second.
15. *pgin* - Number of page-in requests per second.
16. *ppgin* - Number of pages paged in per second.
17. *pflt* - Number of page faults caused by protection errors (copy-on-writes).
18. *vflt* - Number of page faults caused by address translation.
19. *runqsz* - Process run queue size.
20. *freemem* - Number of memory pages available to user processes.
21. *freeswap* - Number of disk blocks available for page swapping.
22. *usr* - Portion of time (%) that cpus run in user mode.
====
Target Variable: usr (numeric, 56 distinct): ['90', '91', '92', '94', '93', '97', '96', '95', '88', '98']
====
Features:

lread (numeric, 235 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
lwrite (numeric, 189 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '11.0']
scall (numeric, 4115 distinct): ['158.0', '220.0', '166.0', '310.0', '160.0', '230.0', '201.0', '195.0', '261.0', '419.0']
sread (numeric, 794 distinct): ['16.0', '10.0', '43.0', '12.0', '95.0', '146.0', '176.0', '13.0', '109.0', '148.0']
swrite (numeric, 640 distinct): ['30.0', '91.0', '24.0', '118.0', '45.0', '22.0', '15.0', '68.0', '62.0', '67.0']
fork (numeric, 228 distinct): ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.8', '2.0', '1.6']
exec (numeric, 386 distinct): ['0.2', '0.4', '0.6', '0.8', '1.0', '1.8', '2.0', '1.2', '2.2', '1.4']
rchar (numeric, 7997 distinct): ['452.0', '6994.0', '7001.0', '425.0', '7018.0', '7026.0', '7007.0', '439.0', '431.0', '416.0']
wchar (numeric, 7939 distinct): ['18709.0', '26776.0', '25473.0', '8482.0', '21962.0', '13554.0', '26377.0', '15749.0', '52096.0', '30990.0']
pgout (numeric, 404 distinct): ['0.0', '0.4', '0.2', '0.6', '0.8', '1.0', '1.6', '1.4', '1.2', '1.8']
ppgout (numeric, 774 distinct): ['0.0', '0.4', '0.6', '0.2', '1.2', '0.8', '1.0', '1.6', '1.8', '1.4']
pgfree (numeric, 1070 distinct): ['0.0', '0.4', '0.6', '0.2', '0.8', '1.2', '1.6', '1.0', '1.4', '1.8']
pgscan (numeric, 1202 distinct): ['0.0', '0.6', '27.0', '1.2', '2.4', '47.4', '2.2', '9.6', '0.8', '27.4']
atch (numeric, 253 distinct): ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8']
pgin (numeric, 832 distinct): ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8']
ppgin (numeric, 1072 distinct): ['0.0', '0.2', '0.4', '0.8', '0.6', '1.0', '1.2', '1.6', '1.4', '1.8']
pflt (numeric, 2987 distinct): ['15.6', '15.8', '15.4', '16.0', '15.57', '15.63', '16.2', '15.2', '15.77', '20.8']
vflt (numeric, 3799 distinct): ['16.8', '17.0', '16.83', '16.77', '17.2', '17.4', '17.6', '18.0', '17.8', '18.2']
runqsz (numeric, 302 distinct): ['1.0', '2.0', '2.2', '1.2', '3.0', '2.4', '1.4', '1.5', '1.6', '1.8']
freemem (numeric, 3165 distinct): ['132.0', '159.0', '168.0', '136.0', '139.0', '133.0', '152.0', '181.0', '143.0', '137.0']
freeswap (numeric, 7658 distinct): ['11.0', '10.0', '9.0', '12.0', '7.0', '13.0', '5.0', '8.0', '6.0', '16.0']
'''

CONTEXT = "CPU Activity System Performance"
TARGET = CuratedTarget(raw_name="usr", new_name="Portion of Time CPUs Run in User Mode",
                       task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="lread", new_name="Reads (Transfers per Second) Between System Memory and User Memory"),
            CuratedFeature(raw_name="lwrite", new_name="Writes (Transfers per Second) Between System Memory and User Memory"),
            CuratedFeature(raw_name="scall", new_name="Number of System Calls of All Types per Second"),
            CuratedFeature(raw_name="sread", new_name="Number of System Read Calls per Second"),
            CuratedFeature(raw_name="swrite", new_name="Number of System Write Calls per Second"),
            CuratedFeature(raw_name="fork", new_name="Number of System Fork Calls per Second"),
            CuratedFeature(raw_name="exec", new_name="Number of System Exec Calls per Second"),
            CuratedFeature(raw_name="rchar", new_name="Number of Characters Transferred per Second by System Read Calls"),
            CuratedFeature(raw_name="wchar", new_name="Number of Characters Transferred per Second by System Write Calls"),
            CuratedFeature(raw_name="pgout", new_name="Number of Page Out Requests per Second"),
            CuratedFeature(raw_name="ppgout", new_name="Number of Pages Paged Out per Second"),
            CuratedFeature(raw_name="pgfree", new_name="Number of Pages per Second Placed on the Free List"),
            CuratedFeature(raw_name="pgscan", new_name="Number of Pages Checked if They Can Be Freed per Second"),
            CuratedFeature(raw_name="atch", new_name="Number of Page Attaches per Second"),
            CuratedFeature(raw_name="pgin", new_name="Number of Page-In Requests per Second"),
            CuratedFeature(raw_name="ppgin", new_name="Number of Pages Paged In per Second"),
            CuratedFeature(raw_name="pflt", new_name="Number of Page Faults Caused by Protection Errors (Copy-on-Writes)"),
            CuratedFeature(raw_name="vflt", new_name="Number of Page Faults Caused by Address Translation"),
            CuratedFeature(raw_name="runqsz", new_name="Process Run Queue Size"),
            CuratedFeature(raw_name="freemem", new_name="Number of Memory Pages Available to User Processes"),
            CuratedFeature(raw_name="freeswap", new_name="Number of Disk Blocks Available for Page Swapping")]
