from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
'''

CONTEXT = "Anonymized Data: Yolanda"
TARGET = CuratedTarget(raw_name="101", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []