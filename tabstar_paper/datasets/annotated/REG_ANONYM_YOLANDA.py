from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
'''

CONTEXT = "Anonymized Data: Yolanda"
TARGET = CuratedTarget(raw_name="101", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []